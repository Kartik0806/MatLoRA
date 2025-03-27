export MODEL_PATH='meta-llama/Llama-3.2-1B'
export SAVE_PATH='.weights/'
export WANDB_DISABLED=true
wandb offline

wget https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json -O ./data/MetaMathQA-395K.json

python3  train_math.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "./data/MetaMathQA-395K.json" \
    --data_length 10000000 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --is_metalora True

python eval_gsm8k.py --model $SAVE_PATH --data_path ./data/test/GSM8K_test.jsonl
python eval_math.py --model $SAVE_PATH --data_path ./data/test/MATH_test.jsonl
