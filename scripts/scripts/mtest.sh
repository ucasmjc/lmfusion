#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export GPU_NUM_PER_NODE=8


export ASCEND_BLOCKING_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=1


export PET_NNODES=1
export PET_NODE_RANK=0
export PET_MASTER_ADDR="localhost"
export PET_MASTER_PORT=16620

nnodes=${PET_NNODES}
node_rank=${PET_NODE_RANK}
nproc_per_node=${GPU_NUM_PER_NODE}
master_addr=${PET_MASTER_ADDR}
master_port=${PET_MASTER_PORT}

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

cd /work/share/projects/mjc/lmfusion

# Log Arguments
export WANDB_PROJECT="lmfusion"
RUN_NAME=t2i
OUTP_DIR=checkpoints/mtest
export WANDB_API_KEY=0f9e4269eec620b5201843ea9fe23e73c8a14b66
export CUDA_LAUNCH_BLOCKING=1
# conda activate janus_pro
torchrun \
 --nproc_per_node=${nproc_per_node} \
 --nnodes=${nnodes} \
 --node_rank=${node_rank} \
 --master_addr=${master_addr} \
 --master_port=${master_port} \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --diffusion_lr 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --report_to tensorboard \
    --run_name $RUN_NAME \
    --sample_rate  0,1,0 \
    --batchsize_list  0,24,0 \
    --samples_per_epoch $[20000*512] \
    --dataset "image_under||image_gen||text_chat" \
    --image_under_data_files  /work/share/projects/zb/datasets/image_under.json \
    --image_under_rootdir /work/share/projects/zb/datasets/image_gen \
    --image_gen_data_files  /work/share/projects/mjc/lmfusion/data/image_gen.json \
    --image_gen_rootdir  /work/share/data/imgdata/aes \
    --text_chat_data_files  /work/share/projects/zb/datasets/text_chat.json \
    --_attn_implementation_new  "sdpa"  


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




 