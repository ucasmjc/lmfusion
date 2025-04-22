#!/bin/bash
# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-1}
# ARG_NPROC_PER_NODE=1
ARG_MASTER_PORT=13600
ARG_RANK=0
# CUDA_VISIBLE_DEVICES=2
# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
MASTER_ADDR="localhost"
MASTER_PORT=$ARG_MASTER_PORT
RANK=$ARG_RANK
cd /work/share/projects/mjc/lmfusion

NODE_RANK=${PET_NODE_RANK}
MASTER_ADDR=${PET_MASTER_ADDR}

# MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"


# Log Arguments
export WANDB_PROJECT="lmfusion"
RUN_NAME=stage1

export WANDB_API_KEY=0f9e4269eec620b5201843ea9fe23e73c8a14b66
OUTP_DIR=checkpoints/stage1
# ===========================================

export CUDA_LAUNCH_BLOCKING=1
# conda activate janus_pro
#--eval_steps 1 \
torchrun --nnodes $NNODES \
    --nproc_per_node $NPROC_PER_NODE \
    --node_rank $NODE_RANK \
    --master_port $ARG_MASTER_PORT\
    train_files/train.py \
    --deepspeed scripts/zero1.json \
    --model_type videollama3_qwen2 \
    --model_path /work/share/projects/mjc/lmfusion/Qwen2.5-1.5B \
    --data_path /work/share1/caption/laion-aes/part0_cap3547693.json \
    --data_folder /work/share/data/imgdata/aes \
    --bf16 False \
    --tf32 False \
    --fp16 True \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --max_steps 250000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --diffusion_lr 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --sample_rate  0,3,0 \
    --batchsize_list  16,16,0 \
    --samples_per_epoch $[500000*384] \
    --dataset "image_under||image_gen||text_chat" \
    --image_under_data_files  /work/share/projects/zb/datasets/image_under.json \
    --image_under_rootdir /work/share/projects/zb/datasets/image_gen \
    --image_gen_data_files  /work/share/projects/mjc/lmfusion/data/image_gen.txt \
    --image_gen_rootdir /work/share/data/imgdata/nolang \
    --text_chat_data_files  /work/share/projects/zb/datasets/text_chat.json \
    --_attn_implementation_new  "sdpa"  \
    #> /work/share/projects/mjc/lmfusion/log/log.txt 2>&1  # 保存所有输出到 log.txt
#--image_gen_rootdir  /work/share1/yqs/uni_dataset/flicker-30k/flickr30k-images \

    #     : bool = field(default=True)
    # is_causal: bool = field(default=False)
    # : bool = field(default=False)


    # 上面的text_chat_data_files 是mini版本
    # --image_under_data_files  /storage/yqs/dataset/BAAI/DenseFusion-1M/DenseFusion-4V-100k/uni_DenseFusion-4V-100k.json \
    # --image_under_rootdir /storage/yqs/dataset/BAAI/DenseFusion-1M/images \
    # --image_gen_data_files  /storage/dataset/filter_aes/cap_merge_final_640/recap2/uni_part0_cap6595998.json \
    # --image_gen_rootdir  /storage/dataset/recap_datacomp_1b_data_20241023_supply/output_undownloaded \
    # --text_chat_data_files /storage/yqs/dataset/BAAI/Infinity-Instruct/uni_7M.json/storage/yqs/dataset/BAAI/Infinity-Instruct/uni_Gen.json 

    # img_und  bs 12  76g  zero1 
    # txet_chat  bs 2 65927MiB  zero1 
    # --image_under_data_files  /storage/yqs/dataset/BAAI/DenseFusion-1M/DenseFusion-4V-100k/mini_uni_DenseFusion-4V-100k.json \
    # --image_under_rootdir /storage/yqs/dataset/BAAI/DenseFusion-1M/images \
    # --image_gen_data_files  /storage/dataset/filter_aes/cap_merge_final_640/recap2/mini_janus_part0_cap6595998.json \
    # --image_gen_rootdir  /storage/dataset/recap_datacomp_1b_data_20241023_supply/output_undownloaded \
    # --text_chat_data_files  /storage/yqs/dataset/BAAI/Infinity-Instruct/mini_uni_Gen.json \




 