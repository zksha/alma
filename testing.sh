# python run_main.py \
#     --rollout_type batched \
#     --meta_model gpt-5 \
#     --execution_model gpt-5-nano/low \
#     --batch_max_update_concurrent 10 \
#     --batch_max_retrieve_concurrent 10 \
#     --task_type alfworld \
#     --status eval_in_distribution \
#     --memo_SHA 53cee295

# python run_main.py \
#     --rollout_type sequential \
#     --meta_model gpt-5 \
#     --execution_model gpt-5-nano/low \
#     --batch_max_update_concurrent 10 \
#     --batch_max_retrieve_concurrent 10 \
#     --task_type alfworld \
#     --status eval_out_of_distribution \
#     --update_task eval_in_distribution \
#     --update_size 70 \
#     --memo_SHA 53cee295

python run_main.py \
    --rollout_type batched \
    --meta_model gpt-5 \
    --execution_model gpt-5-nano/low \
    --batch_max_update_concurrent 10 \
    --batch_max_retrieve_concurrent 10 \
    --task_type textworld \
    --status eval_in_distribution \
    --memo_SHA 70430b60

# python run_main.py \
#     --rollout_type batched \
#     --meta_model gpt-5 \
#     --execution_model gpt-5-nano/low \
#     --batch_max_update_concurrent 10 \
#     --batch_max_retrieve_concurrent 10 \
#     --task_type babaisai \
#     --status eval_in_distribution \
#     --memo_SHA 7e79483e

# python run_main.py \
#     --rollout_type batched \
#     --meta_model gpt-5 \
#     --execution_model gpt-5-nano/low \
#     --batch_max_update_concurrent 10 \
#     --batch_max_retrieve_concurrent 10 \
#     --task_type minihack \
#     --status eval_in_distribution \
#     --memo_SHA 0892408e