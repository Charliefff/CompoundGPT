from typing import Any


from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
from torch.nn import DataParallel
import pandas as pd
import random

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        full_smiles = self.smiles_list[idx]
        full_smiles_with_eos = '<S>' + full_smiles + '<L>'+self.tokenizer.eos_token
        encoding = self.tokenizer.encode_plus(
            full_smiles_with_eos,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=False
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }


class CustomTrainer:
    def __init__(self, df_list, config):

        self.device = config['device']
        self.max_gen_length = config['max_gen_length']
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['tokenizer_name'])
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False)

        if config['checkpoint_path'] is None:
            self.model = GPT2LMHeadModel.from_pretrained(
                config['gpt_size'])
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config['checkpoint_path'])
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            eps=1e-8
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=len(
                df_list) // config['batch_size'] * config['training_epochs']
        )

        training_args = TrainingArguments(
            output_dir=config['model_save_path'],
            num_train_epochs=config['training_epochs'],
            per_device_train_batch_size=config['batch_size'],
            warmup_steps=config['warmup_steps'],
            gradient_accumulation_steps=config['update_loss'],
            weight_decay=0.01,
            logging_dir=config['logdir'],
            logging_steps=config['log_interval'],
            save_steps=config['save_interval'],
            load_best_model_at_end=False,
            evaluation_strategy="no",
            save_strategy="steps",
            fp16=True
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=SMILESDataset(
                df_list, self.tokenizer, config['max_length']),
            data_collator=self.data_collator,
            optimizers=(self.optimizer, self.scheduler),
        )
        self.generate_num = config['generate_num']

    def train(self, reward_model_path="None"):

        self.trainer.train()
        self.save_model(reward_model_path)

    def save_model(self, reward_model_path):
        self.model.save_pretrained(reward_model_path)

    def load_and_generate(self, path, prompt, top_k=50, top_p=0.95, temperature=1, repetition_penalty=3.5, do_sample=True, num_return_sequences=3):
        try:

            self.model = GPT2LMHeadModel.from_pretrained(path)
            self.model.to(self.device)
            self.model.eval()

            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
            encoding = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=128, add_special_tokens=False)
            # print(encoding)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # 生成 SMILES
            output_sequences = self.model.generate(input_ids=input_ids,
                                                   max_length=100,
                                                   eos_token_id=30,
                                                   do_sample=True,
                                                   top_k=top_k,
                                                   top_p=top_p,
                                                   temperature=1.2,
                                                   repetition_penalty=1,
                                                   pad_token_id=0,
                                                   num_return_sequences=self.generate_num)

            generated_smiles = [self.tokenizer.decode(
                g, skip_special_tokens=False) for g in output_sequences]

            return generated_smiles

        except Exception as e:
            print(f"Error during generation: {e}")
            return []

