#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=1
export RANK=0
export ADDR=localhost
export PORT=29501
export NUM_GPUS=8

export OMP_NUM_THREADS=8
export DS_ACCELERATOR=cuda

# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export WANDB_MODE=offline  

PROJECT_ROOT=${PROJECT_ROOT:-"./eventflash"}
MODEL_ROOT=${MODEL_ROOT:-"./models"}
DATA_ROOT=${DATA_ROOT:-"./data"}

LLM_VERSION="${MODEL_ROOT}/Qwen2.5-3B-Instruct"
LLM_BACKBONE="Qwen2"
VISION_MODEL_VERSION="CLIP"
HIDDEN_SIZE=2048
PROMPT_VERSION="eventgpt_qwen"
BASE_RUN_NAME="eventgpt_v2_qwen3b_pretrain"

EVENT_FOLDER="${DATA_ROOT}/TrainingSet"
DATA_PATH="${DATA_ROOT}/pretrain_datasets.json"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    ${PROJECT_ROOT}/train_eventflash.py \
    --deepspeed ${PROJECT_ROOT}/script/ds_config/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --llm_backbone ${LLM_BACKBONE} \
    --event_folder ${EVENT_FOLDER} \
    --event_tower_type ${VISION_MODEL_VERSION} \
    --event_tower ${MODEL_ROOT}/clip-vit-large-patch14 \
    --tuning_target_module="event_projector" \
    --hidden_size ${HIDDEN_SIZE} \
    --event_projector_type mlp \
    --tune_event_projector True \
    --mm_use_ev_start_end False \
    --mm_use_ev_patch_token False \
    --use_npz True \
    --event_size_cfg ${PROJECT_ROOT}/script/event_size_type.yaml \
    --bf16 True \
    --output_dir ${PROJECT_ROOT}/checkpoints/proj_ev_tower/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME