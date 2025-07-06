CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model  Qwen/Qwen3-1.7B\
    --adapters output/v1-20250706-110047/checkpoint-94 \
    --stream true \
    --merge_lora true \
    --infer_backend vllm \
    --max_model_len 8192 \
    --temperature 0 \
    --max_new_tokens 2048
