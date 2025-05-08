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
import torch.nn.functional as F
from tqdm import tqdm
import math
    
class EvalCollator:
    def __init__(
        self,
        item_id2name,
        llm_tokenizer = None,
        llm_model = None,
        device = None,
        prompts = None,
        start_seq_token = None,
        end_seq_token = None,
        item_sep_token = None,
        group_size = 10
    ):

        self.item_id2name = item_id2name
        self.llm_tokenizer = llm_tokenizer
        self.llm_model = llm_model
        self.device = device
        self.systen_message = prompts[0]
        self.instruction = prompts[1]
        self.start_seq_token = start_seq_token
        self.end_seq_token = end_seq_token
        self.item_sep_token = item_sep_token
        self.start_seq_token_id = llm_tokenizer.encode(start_seq_token, add_special_tokens = False)[0]
        self.end_seq_token_id = llm_tokenizer.encode(end_seq_token, add_special_tokens = False)[0]
        self.item_sep_token_id = llm_tokenizer.encode(item_sep_token, add_special_tokens = False)[0]
        self.group_size = group_size

    def __call__(self, batch):
        messages, targets_text = [], []
        for i, example in enumerate(batch):
            seq_name = [self.item_id2name[his] for his in example['seq']]
            input_prompt = self.instruction.replace("[HISTORY_SEQUENCE]", args.start_seq_token+args.item_sep_token.join([t for i, t in enumerate(seq_name)])+args.end_seq_token)

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
            messages.append(message)
            targets_text.append(self.item_id2name[example['next']])

        inputs = self.llm_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding="longest", truncation = False).to(self.device)
        inputs_embeds = self.llm_model.model.embed_tokens(inputs['input_ids'])
        attention_mask = inputs['attention_mask']
        new_inputs_ids, new_inputs_embeds, new_attention_masks, len_list = [], [], [], []
        for i, input_ids in enumerate(inputs['input_ids']):
            start_seq_index, end_seq_index = torch.where(input_ids == self.start_seq_token_id)[0][0].item(), torch.where(input_ids == self.end_seq_token_id)[0][0].item()
            sep_indices = torch.where(input_ids == self.item_sep_token_id)[0].tolist()
            seq_len = len(sep_indices)+1
            new_input_embeds, new_attention_mask = [inputs_embeds[i][:start_seq_index+1]], [attention_mask[i][:start_seq_index+1]]
            item_info = []
            for start_index, end_index in zip([start_seq_index]+sep_indices, sep_indices+[end_seq_index]):
                item_embedding = self.llm_model.model.embed_tokens(input_ids[start_index+1:end_index]).mean(dim=0)
                item_info.append({"embedding":item_embedding, "index":(start_index,end_index)})
            # session-level
            for group_idx in range(math.ceil((seq_len-2*self.group_size) / self.group_size)):
                if seq_len%self.group_size>0:
                    start_item_idx = max(0, (group_idx-1)*self.group_size+seq_len%self.group_size)
                    end_item_idx = group_idx*self.group_size + seq_len%self.group_size
                else:
                    start_item_idx = group_idx*self.group_size
                    end_item_idx = (group_idx+1)*self.group_size
                group_items = item_info[start_item_idx:end_item_idx]
                session_embedding = torch.stack([item["embedding"] for item in group_items]).mean(dim=0)
                new_input_embeds+=[session_embedding.view(1,-1),inputs_embeds[i][group_items[-1]["index"][1]:group_items[-1]["index"][1]+1]]
                new_attention_mask.append(attention_mask[i][group_items[-1]["index"][1]-1:group_items[-1]["index"][1]+1])
            # item-level
            end_item_index = start_seq_index
            for item_idx in range(max(0, seq_len-2*self.group_size), seq_len-self.group_size):
                item = item_info[item_idx]
                end_item_index = item["index"][1]
                new_input_embeds+=[item["embedding"].view(1,-1),inputs_embeds[i][item["index"][1]:item["index"][1]+1]]
                new_attention_mask.append(attention_mask[i][item["index"][1]-1:item["index"][1]+1])
            # text-level
            new_input_embeds.append(inputs_embeds[i][end_item_index+1:])
            new_attention_mask.append(attention_mask[i][end_item_index+1:])
            new_attention_mask = torch.cat(new_attention_mask)
            # partly-compressed prompt
            new_attention_masks.append(new_attention_mask)
            new_inputs_embeds.append(torch.cat(new_input_embeds))
            len_list.append(new_attention_mask.sum().item())
            new_inputs_ids.append(F.pad(input_ids[-10:],(new_attention_mask.sum().item()-10,0),value=-100))
        max_len = max(len_list)
        for i in range(len(len_list)):
            if len(new_attention_masks[i])<max_len:
                new_inputs_embeds[i] = F.pad(new_inputs_embeds[i], (0,0,max_len-new_inputs_embeds[i].size(0),0))
                new_attention_masks[i] = F.pad(new_attention_masks[i], (max_len-new_attention_masks[i].size(0),0))
            else:
                new_inputs_embeds[i] = new_inputs_embeds[i][-max_len:]
                new_attention_masks[i] = new_attention_masks[i][-max_len:]
            if len(new_inputs_ids[i])<max_len:
                new_inputs_ids[i] = F.pad(new_inputs_ids[i], (max_len-new_inputs_ids[i].size(0),0), value=-100)
            else:
                new_inputs_ids[i] = new_inputs_ids[i][-max_len:]
        new_inputs_embeds = torch.stack(new_inputs_embeds)
        new_attention_masks = torch.stack(new_attention_masks)
        new_inputs_ids = torch.stack(new_inputs_ids)
        new_inputs = {'input_ids':new_inputs_ids, 'inputs_embeds':new_inputs_embeds, 'attention_mask':new_attention_masks}

        new_batch={"inputs":new_inputs,
                   "correct_answer": targets_text,
                   }

        return new_batch

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m', type=str, choices=['ml-1m','goodreads','ml-100k'])
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--llm_path', default='/root/model/llama-3.2-1b-instruct', type=str)
    parser.add_argument('--output_dir', default='output/fts', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_return_sequences', type=int, default=20)
    parser.add_argument('--truncate_seq', type=int, default=100)
    parser.add_argument('--group_size', type=int, default=10)
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

    collator = EvalCollator(item_id2title,
                            llm_tokenizer=llm_tokenizer,
                            llm_model=llama_model,
                            device=device,
                            prompts=prompts,
                            group_size=args.group_size,
                            start_seq_token=args.start_seq_token,
                            end_seq_token=args.end_seq_token,
                            item_sep_token=args.item_sep_token)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    prefix_ids = llm_tokenizer("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False).input_ids
    word_ids = [llm_tokenizer(item, add_special_tokens=False).input_ids for item in item_id2title.values()]
    prefix_allowed_tokens_fn = create_prefix_allowed_tokens_fn(candidate_token_ids=word_ids, prefix_token_ids=prefix_ids, tokenizer = llm_tokenizer)
    max_new_tokens = max([len(ids) for ids in word_ids])

    print("prefix ids:", prefix_ids)

    test_content={"generate": [], "real": []}
    def batch_generate(batch):
        generated_ids = llama_model.generate(
            input_ids = batch["inputs"]["input_ids"],
            inputs_embeds=batch["inputs"]["inputs_embeds"],
            attention_mask=batch["inputs"]["attention_mask"],
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
        generated_ids = generated_ids.view(batch["inputs"]["attention_mask"].shape[0], args.num_return_sequences, -1)
        for i, attention_mask in enumerate(batch["inputs"]["attention_mask"]):
            output_ids = [output_ids[len(attention_mask):] for output_ids in generated_ids[i]]
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