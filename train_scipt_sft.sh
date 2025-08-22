export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=warn          # 或 INFO 调试
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# 若 IB 没配好可加：export NCCL_IB_DISABLE=1

torchrun --nproc_per_node=8 --master_port=29501 model_train_ustarvlm_sft.py 2>&1 | tee train_star_vl_qwen2.5_0.5.log