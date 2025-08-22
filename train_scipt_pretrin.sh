export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=warn          # 或 INFO 调试
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
 torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 model_train_ustarvlm_general_pretrain.py 2>&1 | tee train_star_vl_qwen3_4B.log