# ml-1m
# Patch Pretrain
deepspeed --num_gpus=4 pt.py --dataset ml-1m --llm_path your/llama/path --batch_size 16 --gradient_accumulation_steps 1 --learning_rate 8e-6 --truncate_seq 100
# Patch Finetune
# PFT-I
deepspeed --num_gpus=4 fti.py --dataset ml-1m --llm_path your/pt/path --batch_size 16 --gradient_accumulation_steps 1 --learning_rate 8e-6 --truncate_seq 100 --item_len 5
# PFT-S
deepspeed --num_gpus=4 fts.py --dataset ml-1m --llm_path your/pt/path --batch_size 16 --gradient_accumulation_steps 1 --learning_rate 8e-6 --truncate_seq 100 --group_size 5

# goodreads
# Patch Pretrain
deepspeed --num_gpus=4 pt.py --dataset goodreads --llm_path your/llama/path --batch_size 8 --gradient_accumulation_steps 1 --learning_rate 5e-5 --truncate_seq 100
# Patch Finetune
# PFT-I
deepspeed --num_gpus=4 fti.py --dataset goodreads --llm_path your/pt/path --batch_size 8 --gradient_accumulation_steps 1 --learning_rate 5e-5 --truncate_seq 100 --item_len 5
# PFT-S
deepspeed --num_gpus=4 fts.py --dataset goodreads --llm_path your/pt/path --batch_size 8 --gradient_accumulation_steps 1 --learning_rate 5e-5 --truncate_seq 100 --group_size 5

# ml-100k
# Patch Pretrain
deepspeed --num_gpus=4 pt.py --dataset ml-100k --llm_path your/llama/path --batch_size 16 --gradient_accumulation_steps 1 --learning_rate 1e-5 --truncate_seq 100
# Patch Finetune
# PFT-I
deepspeed --num_gpus=4 fti.py --dataset ml-100k --llm_path your/pt/path --batch_size 16 --gradient_accumulation_steps 1 --learning_rate 1e-5 --truncate_seq 100 --item_len 5
# PFT-S
deepspeed --num_gpus=4 fts.py --dataset ml-100k --llm_path your/pt/path --batch_size 16 --gradient_accumulation_steps 1 --learning_rate 3e-6 --truncate_seq 100 --group_size 20