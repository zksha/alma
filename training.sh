python run_main.py \
    --rollout_type batched \
    --meta_model gpt-5 \
    --execution_model gpt-5-nano/low \
    --batch_max_update_concurrent 10 \
    --batch_max_retrieve_concurrent 10 \
    --task_type alfworld \
    --status train \
    --train_size 30

python run_main.py \
    --rollout_type batched \
    --meta_model gpt-5 \
    --execution_model gpt-5-nano/low \
    --batch_max_update_concurrent 10 \
    --batch_max_retrieve_concurrent 10 \
    --task_type textworld \
    --status train 

python run_main.py \
    --rollout_type batched \
    --meta_model gpt-5 \
    --execution_model gpt-5-nano/low \
    --batch_max_update_concurrent 10 \
    --batch_max_retrieve_concurrent 10 \
    --task_type minihack \
    --status train 

python run_main.py \
    --rollout_type batched \
    --meta_model gpt-5 \
    --execution_model gpt-5-nano/low \
    --batch_max_update_concurrent 10 \
    --batch_max_retrieve_concurrent 10 \
    --task_type babaisai \
    --status train 