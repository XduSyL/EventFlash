import torch
import random
import torch.nn as nn
from typing import Optional, List, Union
from transformers.generation.utils import GenerateOutput
from model.eventProjector import build_event_projector, build_point_cloud_projector
from model.eventEncoder import build_event_tower, build_point_cloud_encoder
from utils.token_merge import merge_token
from utils.constents import IGNORE_INDEX, EVENT_TOKEN_INDEX, DEFAULT_EVENT_PATCH_TOKEN, DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN, EVENT_PAD_INDEX
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from model.pc_feature_aggregators import PointFeatureAggregatorSuite
from model.pc_moe_adapter import PCMoEAdapter
import torch.nn.functional as F

class EventGPTv2OutputWrapper:
    def __init__(self, outputs, new_input_ids):
        self.outputs = outputs
        self.new_input_ids = new_input_ids

    def __getattr__(self, item):
        return getattr(self.outputs, item)

    def __getitem__(self, key):
        return self.outputs[key]

    def __iter__(self):
        return iter(self.outputs)

    def keys(self):
        return self.outputs.keys()

class EventGPTv2QwenConfig(Qwen2Config):
    model_type = "eventgpt_v2_qwen" 
    
class EventGPTv2QwenModel(Qwen2Model):
    config_class = EventGPTv2QwenConfig

    def __init__(self, config: Qwen2Config):
        super(EventGPTv2QwenModel, self).__init__(config)

        if hasattr(config, "event_tower"):          
            self.event_tower = build_event_tower(config)
            self.event_projector = build_event_projector(config).to(dtype=torch.bfloat16)
           
    def get_event_tower(self):
        event_tower = getattr(self, 'event_tower', None)
        if type(event_tower) is list:
            event_tower = event_tower[0]
        return event_tower
    

    def initialize_event_modules(self, model_args, fsdp=None):
        event_tower = model_args.event_tower
        self.config.event_tower = event_tower
        
        self.config.use_event_sparsification = getattr(model_args, 'use_event_sparsification', False)
                    
        # Build the event tower
        event_tower = build_event_tower(model_args) 
        self.config.event_tower_type = model_args.event_tower_type
        self.event_tower = event_tower
                          
        # Build the event projector
        event_tower_hidden_size = event_tower.config.hidden_size
        model_args.event_tower_hidden_size = event_tower_hidden_size
        self.event_projector = build_event_projector(model_args).to(dtype=torch.bfloat16)
        self.config.event_tower_hidden_size = event_tower_hidden_size

        # Build density guided compressor
        self.density_compressor = DensityGuidedCompressor(
            input_dim=event_tower_hidden_size,
            hidden_dim=event_tower_hidden_size,
            num_queries=getattr(model_args, 'compressor_queries', 64)
        ).to(dtype=torch.bfloat16)
        
        # setting compress threshold
        self.config.density_threshold = getattr(model_args, 'density_threshold', 0.1)
        
        # Load pretrained weights for visual_projector if provided
        if model_args.pretrain_event_projector is not None:
            print("Loading event_projector pretrain weights...")
            pretrained_weights = torch.load(model_args.pretrain_event_projector)
            # Adjust keys to match model structure
            pretrained_weights = {k.replace("model.event_projector.", ""): v for k, v in pretrained_weights.items()}
            self.event_projector.load_state_dict(pretrained_weights, strict=True)
            print("Pretrained weights loaded successfully into visual_projector.")

def compute_normalized_event_density_batch(event_tensors, patch_size=14):
    """
    Compute normalized event density from event tensors (images).
    Returns a list of density tensors [N_patches].
    """
    density_list = []
    B, C, H, W = event_tensors.shape
    
    for i in range(B):
        img = event_tensors[i] # [C, H, W]
        patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        
        density_map = patches.abs().sum(dim=(0, 3, 4)) # [h_patches, w_patches]
        
        max_density = density_map.max()
        if max_density > 0:
            density_map = density_map / max_density
            
        density_list.append(density_map.flatten()) 

    return density_list


def select_non_white_tokens_batch(event_tensors, event_features, patch_size=14):
    """
    Select tokens corresponding to non-empty patches.
    event_tensors: [B, C, H, W]
    event_features: [B, N_patches, D] (Assuming no CLS token or handled outside)
    Returns: list of tensors [N_kept, D], list of kept indices
    """
    B, C, H, W = event_tensors.shape
    device = event_features.device
    
    kept_features_list = []
    kept_indices_list = []
    
    h_patches = H // patch_size
    w_patches = W // patch_size
    
    for i in range(B):
        img = event_tensors[i] # [C, H, W]
        feat = event_features[i] # [N_patches, D]
        
        # Unfold image to patches [C, h_patches, w_patches, p, p]
        patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.reshape(C, h_patches * w_patches, -1) # [C, N_patches, p*p]
        
        # Check variance or sum to detect "empty"
        # Assuming empty is 0 or constant background (low std dev)
        patch_std = patches.std(dim=(0, 2)) # [N_patches]
        
        # Threshold for "non-white" / "non-empty"
        # Heuristic: std > 1e-4
        non_empty_mask = patch_std > 1e-4
        
        if non_empty_mask.sum() == 0:
            # Keep all if none found (avoid empty tensor)
            non_empty_mask = torch.ones_like(non_empty_mask, dtype=torch.bool)
            
        kept_indices = torch.where(non_empty_mask)[0]
        kept_feat = feat[kept_indices]
        
        kept_features_list.append(kept_feat)
        kept_indices_list.append(kept_indices)
        
    return kept_features_list, kept_indices_list

class DensityGuidedCompressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_queries=64):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

        self.density_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1)  
        )

    def forward(self, token_features, token_densities, attention_mask=None, keep_ratio=None):
        B, N, D = token_features.shape

        K = self.key_proj(token_features)
        V = self.value_proj(token_features)
        Q = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        density_bias = token_densities.unsqueeze(-1) # [B, N, 1]
        density_bias = self.density_encoder(density_bias)
        density_bias = density_bias.transpose(1, 2) # [B, 1, N]
        
        attn_scores = attn_scores + density_bias

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1) == 0, -float('inf'))
            
        attn_weights = F.softmax(attn_scores, dim=-1) # [B, Q, N]
        
        if keep_ratio is not None:
            importance_scores = attn_weights.max(dim=1)[0] # [B, N]
            
            selected_features_list = []
            for i in range(B):
                if attention_mask is not None:
                    valid_len = attention_mask[i].sum().item() # 使用 .item() 转为普通整数
                else:
                    valid_len = N
                    
                k = max(1, int(valid_len * keep_ratio))
                
                scores = importance_scores[i]
                
                if attention_mask is not None:
                    scores = scores.masked_fill(attention_mask[i] == 0, -float('inf'))
                    
                _, topk_indices = torch.topk(scores, k)               
                topk_indices, _ = torch.sort(topk_indices)
                
                selected_feat = token_features[i][topk_indices]
                selected_features_list.append(selected_feat)
                
            return selected_features_list
            
        else:
            return torch.matmul(attn_weights, V)


class EventGPTv2QwenCausalLM(Qwen2ForCausalLM):

    config_class = EventGPTv2QwenConfig

    def __init__(self, config) -> None:
        # super(Qwen2ForCausalLM, self).__init__(config)       
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "EventChat_Qwen"
        config.rope_scaling = None
        self.model = EventGPTv2QwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def get_model(self):
        return self.model
    
    def get_event_tower(self):
        return self.get_model().event_tower
    

    def encode_event(self, event_tensors):
        event_features = self.get_model().get_event_tower()(event_tensors)['last_hidden_state']
        event_features = event_features[:, 1:, :] # 丢弃 CLS token [B, N, D]
        
        use_sparsification = getattr(self.config, 'use_event_sparsification', False)
        
        if not use_sparsification:
            proj_features = self.get_model().event_projector(event_features) # [B, N, D_out]
            final_features_list = [proj_features[i] for i in range(proj_features.shape[0])]
            return final_features_list
            
        else:
            patch_size = self.get_model().get_event_tower().config.patch_size
            
            kept_features_list, kept_indices_list = select_non_white_tokens_batch(
                event_tensors, event_features, patch_size=patch_size
            )
            
            densities = compute_normalized_event_density_batch(event_tensors, patch_size=patch_size)
            
            B = len(kept_features_list)
            D = kept_features_list[0].shape[-1]
            max_len = max([f.shape[0] for f in kept_features_list])
            
            padded_features = torch.zeros((B, max_len, D), device=event_features.device, dtype=event_features.dtype)
            padded_densities = torch.zeros((B, max_len), device=event_features.device, dtype=event_features.dtype)
            attention_mask = torch.zeros((B, max_len), device=event_features.device, dtype=torch.bool)
            
            for i in range(B):
                l = kept_features_list[i].shape[0]
                padded_features[i, :l, :] = kept_features_list[i]
                padded_densities[i, :l] = densities[i][kept_indices_list[i]].to(event_features.dtype)
                attention_mask[i, :l] = True
                
            keep_ratio = getattr(self.config, 'density_threshold', 0.1) 
            
            selected_features_list = self.get_model().density_compressor(
                padded_features, padded_densities, attention_mask=attention_mask, keep_ratio=keep_ratio
            )
            
            final_features_list = []
            for feat in selected_features_list:
                proj_feat = self.get_model().event_projector(feat)
                final_features_list.append(proj_feat)
                
            return final_features_list
    

    def encoder_point_cloud(self, point_cloud):
        emb = self.get_model().embed_tokens.weight
        target_dtype = emb.dtype
        target_device = emb.device

        with torch.cuda.amp.autocast(enabled=False):
            point_cloud_feature = self.get_model().get_point_cloud_encoder()(point_cloud)  

        point_cloud_feature = point_cloud_feature.to(dtype=target_dtype, device=target_device)
        point_cloud_feature = self.get_model().point_cloud_projector(point_cloud_feature)  
        return point_cloud_feature
    
    def build_pc_moe(self, in_dim, out_dim, cfg):
        agg = PointFeatureAggregatorSuite(in_dim, out_dim, voxel_size=cfg.get('voxel_size',0.2), pillar_xy=cfg.get('pillar_xy',0.2), graph_centers=cfg.get('graph_centers',2048), graph_k=cfg.get('graph_k',16))
        moe = PCMoEAdapter(in_dim=out_dim, ctx_dim=out_dim, hidden=cfg.get('router_hidden',128), topk=cfg.get('topk',None), temperature=cfg.get('temperature',1.0))
        return agg, moe

    def encode_point_cloud_moe(self, point_cloud_path, moe_cfg=None):
        emb = self.get_model().embed_tokens.weight
        target_dtype = emb.dtype
        target_device = emb.device

        pt = self.get_model().get_point_cloud_encoder()(point_cloud_path) 

        target_device = self.get_model().embed_tokens.weight.device

        for k, v in list(pt.items()):
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    pt[k] = v.to(device=target_device, dtype=torch.float32, non_blocking=True)
                else:
                    pt[k] = v.to(device=target_device, non_blocking=True)

        agg = self.get_model().pc_agg.to(target_device)
        moe = self.get_model().pc_moe.to(target_device)

        experts = agg.forward_all(pt)              
        ctx = pt["feat"].mean(dim=0, keepdim=True)   

        fused_tokens, w = moe(experts, ctx)
        if fused_tokens is None:
            fused_tokens = pt["feat"]

        fused_tokens = fused_tokens.to(dtype=target_dtype, device=target_device)
        fused_tokens = self.get_model().point_cloud_projector(fused_tokens)
        return fused_tokens
    
    def initialize_event_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_ev_patch_token:
            tokenizer.add_tokens([DEFAULT_EVENT_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_ev_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_event_projector:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_event_projector:
                mm_projector_weights = torch.load(model_args.pretrain_event_projector, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_ev_patch_token:
            if model_args.tune_event_projector:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
                    
    def forward(self, 
            event_tensors: Optional[torch.FloatTensor] = None,
            point_cloud_file: Optional[str] = None,
            input_ids: torch.LongTensor = None, 
            labels: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            event_image_sizes : Optional[List[List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs):

        new_input_ids = None
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                new_input_ids 
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, 
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                event_tensors,
                point_cloud_file,
                event_image_sizes           
            )
            
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )      


        torch.cuda.synchronize()
        return EventGPTv2OutputWrapper(outputs, new_input_ids)
    
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, 
                                            past_key_values, labels, event_tensors,
                                            point_cloud_file=None, event_bin_sizes=None):
        if event_tensors is not None and not isinstance(event_tensors, list):
            event_tensors = [event_tensors]
        if point_cloud_file is not None and not isinstance(point_cloud_file, list):
            point_cloud_file = [point_cloud_file]

        num_patches_per_side = self.get_event_tower().num_patches_per_side
        event_tower = self.get_event_tower()

        if event_tower is None or event_tensors is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        pc_features_list = None
        if point_cloud_file:
            pc_features_list = []
            for pt_file in point_cloud_file:
                pc_feat = self.encoder_point_cloud(pt_file)
                if pc_feat.dim() == 1:
                    pc_feat = pc_feat.unsqueeze(0)
                embed = self.get_model().embed_tokens
                pc_feat = pc_feat.to(device=embed.weight.device, dtype=embed.weight.dtype)
                pc_features_list.append(pc_feat)

        if isinstance(event_tensors, list):
            event_tensors = [x.unsqueeze(0) if x.ndim == 3 else x for x in event_tensors]
        event_tensors_list = []
        for event_tensor in event_tensors:
            event_tensors_list.append(event_tensor if event_tensor.ndim == 4 else event_tensor.unsqueeze(0))

        all_event_tensors = torch.cat([t for t in event_tensors_list], dim=0)
        split_idxs = [t.shape[0] for t in event_tensors_list]
        encoded_all_event_tensors_list = self.encode_event(all_event_tensors)

        event_features = []
        cur_idx = 0
        for split_size in split_idxs:
            cur_ev_feats = encoded_all_event_tensors_list[cur_idx : cur_idx + split_size]
            cur_idx += split_size
            
            if len(cur_ev_feats) > 0:
                ev_feat = torch.cat(cur_ev_feats, dim=0) 
            else:
                embed_dim = self.config.hidden_size 
                dtype = self.get_model().embed_tokens.weight.dtype
                ev_feat = torch.empty((0, embed_dim), device=all_event_tensors.device, dtype=dtype)

            event_features.append(ev_feat)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
        labels = [lab[mask] for lab, mask in zip(labels, attention_mask)]

        new_input_embeds, new_labels, new_input_ids_list = [], [], []
        cur_event_idx = 0
        embed = self.get_model().embed_tokens
        device = embed.weight.device
        dtype = embed.weight.dtype

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_event_bin = int((cur_input_ids == EVENT_TOKEN_INDEX).sum().item())

            if num_event_bin == 0:
                cur_input_embeds_1 = embed(cur_input_ids.to(device))
                new_input_embeds.append(cur_input_embeds_1)
                new_labels.append(labels[batch_idx].to(device))
                new_input_ids_list.append(cur_input_ids.to(device))
                continue

            event_token_pos = torch.where(cur_input_ids == EVENT_TOKEN_INDEX)[0].tolist()
            cur_labels = labels[batch_idx]

            seg_starts = [-1] + event_token_pos + [cur_input_ids.shape[0]]
            text_segments = []
            label_segments = []
            for i in range(len(seg_starts) - 1):
                text_segments.append(cur_input_ids[seg_starts[i] + 1 : seg_starts[i + 1]])
                label_segments.append(cur_labels[seg_starts[i] + 1 : seg_starts[i + 1]])

            text_lens = [seg.shape[0] for seg in text_segments]
            if sum(text_lens) > 0:
                text_all = embed(torch.cat(text_segments).to(device))
                text_splits = torch.split(text_all, text_lens, dim=0)
            else:
                text_splits = [torch.empty((0, embed.embedding_dim), device=device, dtype=dtype) for _ in text_segments]

            cur_new_input_embeds, cur_new_labels, cur_new_input_ids = [], [], []
            inserted_once = False

            for i in range(num_event_bin + 1):
                cur_new_input_embeds.append(text_splits[i])
                cur_new_labels.append(label_segments[i].to(device))
                cur_new_input_ids.append(text_segments[i].to(device))

                if i < num_event_bin:
                    if not inserted_once:
                        ev_feat = event_features[cur_event_idx].to(device=device, dtype=dtype)
                        if pc_features_list is not None and batch_idx < len(pc_features_list) and pc_features_list[batch_idx] is not None:
                            ev_feat = torch.cat([ev_feat, pc_features_list[batch_idx]], dim=0)
                        cur_new_input_embeds.append(ev_feat)
                        cur_new_labels.append(torch.full((ev_feat.shape[0],), IGNORE_INDEX, device=device, dtype=cur_labels.dtype))
                        cur_new_input_ids.append(torch.full((ev_feat.shape[0],), EVENT_PAD_INDEX, device=device, dtype=torch.long))
                        inserted_once = True
                        cur_event_idx += 1
                    else:
                        pass

            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            cur_new_input_ids = torch.cat(cur_new_input_ids, dim=0)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_input_ids_list.append(cur_new_input_ids)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_input_ids_list = [x[:tokenizer_model_max_length] for x in new_input_ids_list]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_input_ids_padded = torch.full((batch_size, max_len), EVENT_PAD_INDEX, dtype=torch.long, device=new_input_ids_list[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (emb_i, lab_i, ids_i) in enumerate(zip(new_input_embeds, new_labels, new_input_ids_list)):
            cur_len = emb_i.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, emb_i.shape[1]), dtype=emb_i.dtype, device=emb_i.device), emb_i), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = lab_i
                    new_input_ids_padded[i, -cur_len:] = ids_i
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((emb_i, torch.zeros((max_len - cur_len, emb_i.shape[1]), dtype=emb_i.dtype, device=emb_i.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = lab_i
                    new_input_ids_padded[i, :cur_len] = ids_i
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_input_ids_padded
  
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        event_tensors: Optional[torch.Tensor] = None,
        event_image_sizes: Optional[torch.Tensor] = None,
        event_data=None,
        event_feature = None,
        point_cloud_file = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if event_tensors is not None or event_data is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                new_input_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                event_tensors,
                point_cloud_file
            )
        else:
            raise NotImplementedError("please input Event")
        
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        event_tensors = kwargs.pop("event_tensors", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if event_tensors is not None:
            inputs['event_tensors'] = event_tensors
        return inputs
    
              
AutoConfig.register("eventgpt_v2_qwen", EventGPTv2QwenConfig)
AutoModelForCausalLM.register(EventGPTv2QwenConfig, EventGPTv2QwenCausalLM)