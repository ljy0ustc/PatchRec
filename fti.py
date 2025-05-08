import os
import torch
import re
import pandas as pd
import os.path as op
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk
from trl import SFTConfig, DataCollatorForCompletionOnlyLM
from fti_trainer import FTITrainer
from accelerate import Accelerator
from utils import get_id2title, get_data, print_parameters_info, set_seed, get_prompt
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--output_dir', type=str, default="output/ft-i")
    parser.add_argument('--logging_dir', type=str, default="logs/ft-i")
    parser.add_argument('--llm_path', type=str, default="/root/model/llama-3.2-1b-instruct")
    parser.add_argument('--ds_config_path', type=str, default="./deepspeed/ds_z2_config.json")
    parser.add_argument('--dataset', type=str, default='ml-1m', choices=['ml-1m','goodreads','ml-100k'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--cutoff_len', type=int, default=32768)
    parser.add_argument('--truncate_seq', type=int, default=100)
    parser.add_argument('--item_len', type=int, default=10)
    parser.add_argument('--eval_step', type=float, default=0.2)
    parser.add_argument('--save_step', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--start_seq_token', type=str, default='<|start_seq_id|>')
    parser.add_argument('--end_seq_token', type=str, default='<|end_seq_id|>')
    parser.add_argument('--item_sep_token', type=str, default='<|item_sep_id|>')
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    return args

def train(args):

    args.data_dir = op.join(args.data_dir, args.dataset)
    args.output_dir = op.join(args.output_dir, args.dataset, f"sft-epoch{args.num_train_epochs}-sl{args.truncate_seq}-il{args.item_len}-bs{args.batch_size}-gas{args.gradient_accumulation_steps}-lr{args.learning_rate}")
    args.logging_dir = op.join(args.logging_dir, args.dataset, f"sft-epoch{args.num_train_epochs}-sl{args.truncate_seq}-il{args.item_len}-bs{args.batch_size}-gas{args.gradient_accumulation_steps}-lr{args.learning_rate}")

    if op.exists(op.join(args.data_dir, "cache", f"train_data_seqlen-{args.truncate_seq}")):
        train_data = load_from_disk(op.join(args.data_dir, "cache", f"train_data_seqlen-{args.truncate_seq}"))
        val_data = load_from_disk(op.join(args.data_dir, "cache", f"val_data_seqlen-{args.truncate_seq}"))
    else:
        data_path = op.join(args.data_dir, "seq.df")
        data_files = {
            "train": get_data(data_path, "train", args.truncate_seq),
            "val": get_data(data_path, "val", args.truncate_seq),
        }
        train_data = Dataset.from_pandas(data_files["train"]).shuffle(seed=args.seed)
        val_data = Dataset.from_pandas(data_files["val"]).shuffle(seed=args.seed)
        train_data.save_to_disk(op.join(args.data_dir, "cache", f"train_data_seqlen-{args.truncate_seq}"))
        val_data.save_to_disk(op.join(args.data_dir, "cache", f"val_data_seqlen-{args.truncate_seq}"))

    print("len of train_data:",len(train_data))
    print("len of val data:", len(val_data))

    llama_model = AutoModelForCausalLM.from_pretrained(args.llm_path, torch_dtype=torch.bfloat16, use_cache=False, attn_implementation="flash_attention_2")
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    llm_tokenizer.add_special_tokens({'additional_special_tokens': [args.start_seq_token,args.end_seq_token,args.item_sep_token]})
    llama_model.resize_token_embeddings(len(llm_tokenizer))
    llm_tokenizer.padding_side = "right"  
    llm_tokenizer.pad_token_id = (0)
        
    item_id2title = get_id2title(args.data_dir)
    prompts = get_prompt(op.join(args.data_dir, "prompt.txt"))
    template = {
        "system": prompts[0],
        "instruction": prompts[1]
    }
    
    def formatting_func(examples):
        processed_examples = []
        for i in range(len(examples['seq'])):
            seq = examples['seq'][i]
            next_item = examples['next'][i]
            seq_title = [item_id2title[his] for his in seq]
            input_prompt = template["instruction"].replace("[HISTORY_SEQUENCE]", args.start_seq_token+args.item_sep_token.join([t for t in seq_title])+args.end_seq_token)
        
            message = [
                {
                    "role": "system",
                    "content": template["system"]
                },
                {
                    "role": "user",
                    "content": input_prompt
                },
                {
                    "role": "assistant",
                    "content": item_id2title[next_item]
                }
            ]
            new_example = llm_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            processed_examples.append(new_example)
        return processed_examples
    
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(llm_tokenizer.encode(response_template, add_special_tokens = False), tokenizer=llm_tokenizer)

    training_args = SFTConfig(
        max_seq_length=args.cutoff_len,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=args.num_train_epochs, 
        learning_rate=args.learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=args.save_step,
        save_total_limit=100,
        load_best_model_at_end=False,
        eval_strategy="steps",
        eval_steps=args.eval_step,
        logging_steps=1,
        output_dir=args.output_dir,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to=None,
        logging_dir=args.logging_dir,
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        save_only_model=True,
        deepspeed=args.ds_config_path,
        ddp_timeout=180000000,
    )

    print("Begin Training ... ")
    print_parameters_info(llama_model)

    trainer = FTITrainer(
        llama_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        tokenizer=llm_tokenizer,
        formatting_func=formatting_func,
        training_args=training_args,
        start_seq_token=llm_tokenizer.encode(args.start_seq_token, add_special_tokens = False)[0],
        end_seq_token=llm_tokenizer.encode(args.end_seq_token, add_special_tokens = False)[0],
        item_sep_token=llm_tokenizer.encode(args.item_sep_token, add_special_tokens = False)[0],
        item_len=args.item_len
    )

    trainer.train() 
    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    args = parser()
    set_seed(args.seed)
    train(args)