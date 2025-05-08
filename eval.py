from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import os
import os.path as op
from utils import get_id2title, set_seed, get_prompt, create_prefix_allowed_tokens_fn, get_data
from calculate_metrics import compute_hr, compute_ndcg
from pandas.core.frame import DataFrame
import argparse
from tqdm import tqdm
    
class EvalCollator:
    def __init__(
        self,
        item_id2name,
        llm_tokenizer = None,
        device = None,
        prompts = None,
        seq_sep = '<|item_sep_id|>'
    ):

        self.item_id2name = item_id2name
        self.seq_sep = seq_sep
        self.llm_tokenizer = llm_tokenizer
        self.device = device
        self.systen_message = prompts[0]
        self.instruction = prompts[1]

    def __call__(self, batch):
        messages, targets_text = [], []
        for i, example in enumerate(batch):
            seq_name = [self.item_id2name[his] for his in example['seq']]
            input_prompt = self.instruction.replace("[HISTORY_SEQUENCE]", self.seq_sep.join([f"<{t}>" for i, t in enumerate(seq_name)]))

            message = [
                {
                    "role": "system",
                    "content": self.systen_message
                },
                {
                    "role": "user",
                    "content": input_prompt
                }
            ]
            #print(message)
            messages.append(message)
            targets_text.append(self.item_id2name[example['next']])

        batch_tokens = self.llm_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding="longest", truncation = False).to(self.device)
        new_batch={"test_tokens":batch_tokens,
                   "correct_answer": targets_text,
                   }

        return new_batch

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m', type=str, choices=['ml-1m','goodreads','ml-100k'])
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--llm_path', default='/root/model/llama-3.2-1b-instruct', type=str)
    parser.add_argument('--output_dir', default='output/vanilla', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_return_sequences', type=int, default=20)
    parser.add_argument('--truncate_seq', type=int, default=100)
    parser.add_argument('--start_seq_token', type=str, default='<|start_seq_id|>')
    parser.add_argument('--end_seq_token', type=str, default='<|end_seq_id|>')
    parser.add_argument('--item_sep_token', type=str, default='<|item_sep_id|>')
    args = parser.parse_args()
    return args

def calculate_rank(eval_content, candidates_title):
    pred_ranks = []
    for i,generate_list in enumerate(eval_content["generate"]):
        real = eval_content["real"][i]
        pred_rank = 20
        for rank, generate_item in enumerate(generate_list):
            if real == generate_item[:len(real)]:
                pred_rank = rank
        pred_ranks.append(pred_rank)
    return pred_ranks

def evaluate(args):
    args.output_dir = op.join(args.output_dir, args.dataset, "test")
    args.data_dir = op.join(args.data_dir, args.dataset)
    
    accelerator = Accelerator()
    device = accelerator.device

    llama_model = AutoModelForCausalLM.from_pretrained(args.llm_path, torch_dtype=torch.bfloat16, use_cache=False, attn_implementation="flash_attention_2").to(device)

    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    llm_tokenizer.padding_side = "left"  
    llm_tokenizer.pad_token_id = (0)

    llama_model.eval()

    print("Model is on device:", llama_model.device)
        
    item_id2title = get_id2title(args.data_dir)
    data_files = {
        "test": get_data(op.join(args.data_dir, "seq.df"), "test", args.truncate_seq),
    }

    test_data = Dataset.from_pandas(data_files["test"]).shuffle(seed=args.seed)
    prompts = get_prompt(op.join(args.data_dir, "prompt.txt"))

    collator = EvalCollator(item_id2title, llm_tokenizer=llm_tokenizer, device=device, prompts=prompts, seq_sep=args.item_sep_token)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    prefix_ids = llm_tokenizer("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False).input_ids
    word_ids = [llm_tokenizer(item, add_special_tokens=False).input_ids for item in item_id2title.values()]
    prefix_allowed_tokens_fn = create_prefix_allowed_tokens_fn(candidate_token_ids=word_ids, prefix_token_ids=prefix_ids, tokenizer = llm_tokenizer)
    max_new_tokens = max([len(ids) for ids in word_ids])

    print("prefix ids:", prefix_ids)

    test_content={"generate": [], "real": []}
    def batch_generate(batch):
        generated_ids = llama_model.generate(
            input_ids = batch["test_tokens"],
            max_new_tokens=max_new_tokens,
            num_beams = args.num_return_sequences,
            num_beam_groups = args.num_return_sequences,
            diversity_penalty = 0.8,
            num_return_sequences = args.num_return_sequences,
            prefix_allowed_tokens_fn = prefix_allowed_tokens_fn,
            do_sample = False,
            temperature = 0.6,
            top_p = 0.9,
            top_k = 20,
            early_stopping = True,
            eos_token_id = llm_tokenizer.eos_token_id,
            pad_token_id = llm_tokenizer.pad_token_id,
            use_cache = False
        )
        outputs = []
        generated_ids = generated_ids.view(batch["test_tokens"].shape[0], args.num_return_sequences, -1)
        for i, input_ids in enumerate(batch["test_tokens"]):
            output_ids = [output_ids[len(input_ids):] for output_ids in generated_ids[i]]
            output_text = llm_tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            output = [text.strip() for text in output_text]
            outputs.append(output)
        return outputs

    for batch in tqdm(test_dataloader,desc="Predicting Batches"):
        batch = accelerator.prepare(batch)
        generate_output = batch_generate(batch)
        for i, generate in enumerate(generate_output):
            test_content["generate"].append(generate)
            test_content["real"].append(batch['correct_answer'][i])

    df=DataFrame(test_content)
    if not op.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df.to_csv(op.join(args.output_dir, 'test.csv'))
    pred_ranks = calculate_rank(test_content, item_id2title.values())

    print("load checkpoint from:", args.llm_path)
    hr = compute_hr(pred_ranks, args.num_return_sequences)
    ndcg = compute_ndcg(pred_ranks, args.num_return_sequences)
    print(f"hr@{args.num_return_sequences}:", hr)
    print(f"ndcg@{args.num_return_sequences}:", ndcg)

if __name__ == "__main__":
    args = parser()
    set_seed(args.seed)
    evaluate(args)