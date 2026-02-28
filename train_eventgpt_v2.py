import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers
from transformers import AutoTokenizer
from model.eventgptv2_llama import EventGPTv2LLaMACausalLM
from model.eventgptv2_qwen import EventGPTv2QwenCausalLM
from dataset import conversation as conversation_lib
from dataset.EventChatDataset import make_supervised_data_module
from argument import ModelArguments, DataArguments, TrainingArguments
from model.event_trainer import EventChatTrainer, compute_metrics, safe_save_model_for_hf_trainer, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from swanlab.integration.transformers import SwanLabCallback
import pathlib

def get_base_model(model_args: ModelArguments, training_args: TrainingArguments):
    if model_args.llm_backbone == 'llama':
        model = EventGPTv2LLaMACausalLM.from_pretrained(model_args.model_name_or_path,
                                            cache_dir=training_args.cache_dir,
                                            attn_implementation="flash_attention_2",
                                            torch_dtype=(torch.bfloat16 if training_args.bf16 else None))
        
    elif model_args.llm_backbone == 'Qwen2':
        model = EventGPTv2QwenCausalLM.from_pretrained(model_args.model_name_or_path,
                                            cache_dir=training_args.cache_dir,
                                            attn_implementation="flash_attention_2",
                                            torch_dtype=(torch.bfloat16 if training_args.bf16 else None))
    else: 
        raise ValueError(f"Model {model_args.model_name_or_path} not supported")
    return model

if __name__ == '__main__':
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses() 
    
    local_rank = training_args.local_rank
    model = get_base_model(model_args, training_args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        )

    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        raise ValueError(f"Conversation template {model_args.version} not found")
        
    model.get_model().initialize_event_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )

    event_tower = model.get_event_tower()
    event_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    
    data_args.event_processor = event_tower.event_processor
    data_args.is_multimodal = True
    
    if model_args.llm_backbone == 'Qwen2':
        tokenizer.padding_side = 'left'
    
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    
    
    if model_args.tune_event_projector:
        training_args.tune_event_projector = model_args.tune_event_projector
        model.requires_grad_(False)
        for p in model.get_model().event_projector.parameters():
            p.requires_grad = True

    if model_args.tuning_target_module is not None:
        model.config.tuning_target_module = training_args.tuning_target_module = model_args.tuning_target_module
        model.requires_grad_(False)
        event_tower.requires_grad_(False)
        model.get_model().event_projector.requires_grad_(False)
        tuning_target_module = model_args.tuning_target_module.split(',')
        if 'event_tower' in tuning_target_module:
            for name,param in model.named_parameters():
                if 'event_tower' in name:
                    param.requires_grad_(True)
                
        if 'event_projector' in tuning_target_module:
            for p in model.get_model().event_projector.parameters():
                p.requires_grad = True
            
        if 'llm_backbone' in tuning_target_module:
            exclude_keys = (
                'event_projector',
                'event_tower',
                'point_cloud_projector',
                'point_cloud_encoder',
            )
            for name, p in model.named_parameters():
                if not any(k in name for k in exclude_keys):
                    p.requires_grad_(True)

        
        if 'point_cloud_projector' in tuning_target_module:
            for p in model.get_model().point_cloud_projector.parameters():
                p.requires_grad = True 
                
        if 'point_encoder' in tuning_target_module:
            for p in model.get_model().point_cloud_encoder.parameters():
                p.requires_grad = True                  
        
        if 'all' in tuning_target_module:
            for name,param in model.named_parameters():
                param.requires_grad_(True)
                
    total_params = sum(p.ds_numel if hasattr(p, 'ds_numel') else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, 'ds_numel') else p.numel() for p in model.parameters() if p.requires_grad)   
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    model.config.mm_use_ev_start_end = data_args.mm_use_ev_start_end = model_args.mm_use_ev_start_end
    model.config.event_projector_type = data_args.event_projector_type = model_args.event_projector_type
    model.config.mm_use_ev_patch_token = data_args.mm_use_ev_patch_token = model_args.mm_use_ev_patch_token
    
    model.initialize_event_tokenizer(model_args, tokenizer=tokenizer)
          
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    trainer = EventChatTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        train_dataset=data_module['train_dataset'],
        data_collator=data_module['data_collator'],
        callbacks=[SwanLabCallback(project="EventGPT-V2")]
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()

    model.config.use_cache = True

    if training_args.useLora:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
        
    print(f"Model saved to {training_args.output_dir}")


    
    
    
    
    