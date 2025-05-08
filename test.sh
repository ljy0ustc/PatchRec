# ml-1m
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset ml-1m --llm_path ckpt/ml-1m/SFT  --batch_size 16
CUDA_VISIBLE_DEVICES=1 python fti_eval.py --dataset ml-1m --llm_path ckpt/ml-1m/PPT --batch_size 16 --item_len 5
CUDA_VISIBLE_DEVICES=2 python fti_eval.py --dataset ml-1m --llm_path ckpt/ml-1m/PFT-I  --batch_size 16 --item_len 5
CUDA_VISIBLE_DEVICES=3 python fts_eval.py --dataset ml-1m --llm_path ckpt/ml-1m/PFT-S --batch_size 16 --group_size 5

# goodreads
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset goodreads --llm_path ckpt/goodreads/SFT  --batch_size 16
CUDA_VISIBLE_DEVICES=1 python fti_eval.py --dataset goodreads --llm_path ckpt/goodreads/PPT --batch_size 16 --item_len 5
CUDA_VISIBLE_DEVICES=2 python fti_eval.py --dataset goodreads --llm_path ckpt/goodreads/PFT-I  --batch_size 16 --item_len 5
CUDA_VISIBLE_DEVICES=3 python fts_eval.py --dataset goodreads --llm_path ckpt/goodreads/PFT-S --batch_size 16 --group_size 5

# ml-100k
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset ml-100k --llm_path ckpt/ml-100k/SFT  --batch_size 16
CUDA_VISIBLE_DEVICES=1 python fti_eval.py --dataset ml-100k --llm_path ckpt/ml-100k/PPT  --batch_size 16 --item_len 5
CUDA_VISIBLE_DEVICES=2 python fti_eval.py --dataset ml-100k --llm_path ckpt/ml-100k/PFT-I  --batch_size 16 --item_len 5
CUDA_VISIBLE_DEVICES=3 python fts_eval.py --dataset ml-100k --llm_path ckpt/ml-100k/PFT-S  --batch_size 16 --group_size 20