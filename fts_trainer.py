import torch
from trl import SFTTrainer
from packaging import version
from transformers.utils import is_peft_available
import importlib.metadata
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import torch.nn.functional as F
import random
import math

if is_peft_available():
    from peft import PeftModel

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

class FTSTrainer(SFTTrainer):
    def __init__(self, model=None, train_dataset=None, eval_dataset=None, data_collator=None, tokenizer=None, formatting_func=None, training_args=None, start_seq_token=None, end_seq_token=None, item_sep_token=None, group_size=10):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            formatting_func=formatting_func,
            args=training_args
        )
        self.start_seq_token = start_seq_token
        self.end_seq_token = end_seq_token
        self.item_sep_token = item_sep_token
        self.tokenizer = tokenizer
        self.model_accepts_loss_kwargs = True
        self.group_size = group_size
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if hasattr(model, 'module'):
            inputs_embeds = model.module.model.embed_tokens(inputs['input_ids'])
        else:
            inputs_embeds = model.model.embed_tokens(inputs['input_ids'])
        attention_mask, labels = inputs['attention_mask'], inputs['labels']
        new_inputs_embeds, new_attention_masks, new_labels, len_list = [], [], [], []
        for i, input_ids in enumerate(inputs['input_ids']):
            start_seq_index, end_seq_index = torch.where(input_ids == self.start_seq_token)[0][0].item(), torch.where(input_ids == self.end_seq_token)[0][0].item()
            sep_indices = torch.where(input_ids == self.item_sep_token)[0].tolist()
            seq_len = len(sep_indices)+1
            new_input_embeds, new_attention_mask, new_label = [inputs_embeds[i][:start_seq_index+1]], [attention_mask[i][:start_seq_index+1]], [labels[i][:start_seq_index+1]]
            item_info = []
            for start_index, end_index in zip([start_seq_index]+sep_indices, sep_indices+[end_seq_index]):
                if hasattr(model, 'module'):
                    item_embedding = model.module.model.embed_tokens(input_ids[start_index+1:end_index]).mean(dim=0)
                else:
                    item_embedding = model.model.embed_tokens(input_ids[start_index+1:end_index]).mean(dim=0)
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
                new_label.append(labels[i][group_items[-1]["index"][1]-1:group_items[-1]["index"][1]+1])
            # item-level
            end_item_index = start_seq_index
            for item_idx in range(max(0, seq_len-2*self.group_size), seq_len-self.group_size):
                item = item_info[item_idx]
                end_item_index = item["index"][1]
                new_input_embeds+=[item["embedding"].view(1,-1),inputs_embeds[i][item["index"][1]:item["index"][1]+1]]
                new_attention_mask.append(attention_mask[i][item["index"][1]-1:item["index"][1]+1])
                new_label.append(labels[i][item["index"][1]-1:item["index"][1]+1])
            # text-level
            new_input_embeds.append(inputs_embeds[i][end_item_index+1:])
            new_attention_mask.append(attention_mask[i][end_item_index+1:])
            new_label.append(labels[i][end_item_index+1:])
            new_attention_mask = torch.cat(new_attention_mask)
            # partly-compressed prompt
            new_attention_masks.append(new_attention_mask)
            new_inputs_embeds.append(torch.cat(new_input_embeds))
            new_labels.append(torch.cat(new_label))
            len_list.append(new_attention_mask.sum().item())
        max_len = max(len_list)
        for i in range(len(len_list)):
            if len(new_attention_masks[i])<max_len:
                new_inputs_embeds[i] = F.pad(new_inputs_embeds[i], (0,0,0,max_len-new_inputs_embeds[i].size(0)))
                new_attention_masks[i] = F.pad(new_attention_masks[i], (0,max_len-new_attention_masks[i].size(0)))
                new_labels[i] = F.pad(new_labels[i], (0,max_len-new_labels[i].size(0)), value=-100)
            else:
                new_inputs_embeds[i] = new_inputs_embeds[i][:max_len]
                new_attention_masks[i] = new_attention_masks[i][:max_len]
                new_labels[i] = new_labels[i][:max_len]
        new_inputs_embeds = torch.stack(new_inputs_embeds)
        new_attention_masks = torch.stack(new_attention_masks)
        new_labels = torch.stack(new_labels)
        new_inputs = {'inputs_embeds':new_inputs_embeds, 'attention_mask':new_attention_masks, 'labels':new_labels}
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in new_inputs:
            labels = new_inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            new_inputs = {**new_inputs, **loss_kwargs}
        outputs = model(**new_inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(new_inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss