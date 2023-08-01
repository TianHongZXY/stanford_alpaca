#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from transformers.optimization import get_scheduler
import utils
from torch.utils.data import Dataset, DataLoader
# from transformers import Trainer
import lightning.pytorch as pl
# from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
# from pysnooper import snoop


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)
        # list_data_dict = list_data_dict[:1]

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class GPTLightning(pl.LightningModule):
    def __init__(self, args=None, model=None, tokenizer=None):
        super().__init__()
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
            self.save_hyperparameters(args)
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    # def setup(self, stage) -> None:
    #     if stage == 'fit':
    #         # train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
    
    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def on_train_epoch_end(self):
        if self.global_rank == 0:
            self.save_hf_checkpoint()

    def save_hf_checkpoint(self) -> None:
        """Save huggingface model checkpoint and tokenizer"""
        #  if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
        save_path = os.path.join(
            self.trainer.checkpoint_callback.dirpath if self.trainer else self.hparams.save_path,
            'hf_pretrained_epoch{}_step{}'.format(self.current_epoch, self.global_step))
        state_dict = self.model.state_dict()
        self.model.save_pretrained(
            save_path, state_dict=state_dict, safe_serialization=False
        )
        # self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        # Good practice: save your training arguments together with the trained model
        with open(os.path.join(save_path, "training_args.json"), "w") as f:
            json.dump(dict(self.hparams), f, indent=4)

    # def save_hf_model(self):
    #     save_path = "./outputs" # self.hparams.save_path
    #     state_dict = self.model.state_dict()
    #     self.model.save_pretrained(
    #         save_path, state_dict=state_dict, safe_serialization=False
    #     )
    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(save_path)
    #     # Good practice: save your training arguments together with the trained model
    #     with open(os.path.join(save_path, "training_args.json"), "w") as f:
    #         json.dump(dict(self.hparams), f, indent=4)
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {'params': [p for n, p in self.trainer.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.hparams.l2},
            {'params': [p for n, p in self.trainer.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = getattr(torch.optim, self.hparams.optimizer)(optimizer_grouped_params, lr=self.hparams.lr)

        # Configure learning rate scheduler.
        warmup_steps = self.hparams.warmup_ratio * self.hparams.total_step
        scheduler = get_scheduler(name=self.hparams.scheduler, optimizer=optimizer,
                                  num_warmup_steps=warmup_steps, num_training_steps=self.hparams.total_step)
        lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--l2", type=float, default=0.)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    args = parser.parse_args()
    MAX_EPOCHS = 3

    gpt = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=gpt,
    )
    # gpt.gradient_checkpointing_enable()

    collate_fn = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.micro_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    args.total_step = len(train_dataset) * MAX_EPOCHS // args.micro_batch_size // 2 // 16
    model = GPTLightning(args, gpt)
    # Calculate total steps
    # world_size = 1 # trainer.world_size
    # tb_size = args.micro_batch_size * max(1, world_size)
    # ab_size = 1 # accumulate_grad_batches
    # print(f"Training batch size: {tb_size * ab_size}")
    #     total_step = (len(train_loader.dataset) * 
    #                     self.trainer.max_epochs // tb_size) // ab_size
    # else:
    #     self.total_step = self.trainer.max_steps // self.trainer.accumulate_grad_batches

    # print('Total training step:', self.total_step)

    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # strategy = DDPStrategy(find_unused_parameters=False)
    # sd = torch.load("./lightning_logs/version_3/checkpoints/epoch=2-step=4878.ckpt")
    # model.load_state_dict(sd['state_dict'])
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=2,
        max_epochs=3,
        accumulate_grad_batches=16,
        precision="bf16-mixed",
        logger=True,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        enable_checkpointing=True
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
    model.save_hf_checkpoint()


if __name__ == "__main__":
    # print(torch.cuda.device_count())
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42)
    train()
