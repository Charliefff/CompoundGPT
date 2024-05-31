from trainer import SMILESDataset, CustomTrainer
from rich import print
import pandas as pd
from tqdm import tqdm
import torch
import random
import os


def read_zinc20(path, sample_size):
    zinc20_dir = os.listdir(path)
    zinc20_dir_sample = random.sample(zinc20_dir, sample_size)
    zinc20_list = []
    print('Loading data...')
    for i in tqdm(zinc20_dir_sample):
        df = pd.read_parquet(path+i)
        df_list = df['smiles'].tolist()
        zinc20_list.extend(df_list)

    return zinc20_list

def read_pubchem(path):
    df = pd.read_csv(path)
    df_list = df['smiles'].tolist()
    return df_list

def main():
    if not config['generate']:
        zinc20 = '/data/tzeshinchen/research/dataset/zinc20/'
        num_samples = 20
        df_list = read_zinc20(zinc20, num_samples)
        df_list = list(set(df_list))
        print('Number of samples: ', len(df_list))
        print()
        print('Start training...')
        trainer = CustomTrainer(df_list, config)
        trainer.train()

    if config['generate']:
        print('Start generating...')
        config['logdir'] = 'logs/Compound_GPT2_Demo'
        df_list = ['C']
        trainer = CustomTrainer(df_list, config)
        output = trainer.load_and_generate(
            '/data/tzeshinchen/research/gpt2_hugginface/model_save_path_zinc20/pretrain/checkpoint-190000', 
            '<S>')
        with open('./output/output.txt', 'w') as f:
            for i in output:
                f.write(i+'\n')

if __name__ == '__main__':
    
    config = {
        "dataset": "zinc20",
        "gpt_size": 'gpt2',
        "tokenizer_name": './zinc20M_gpt2_tokenizer',
        "learning_rate": 5e-5,
        "batch_size": 256,
        "update_loss": 1,
        "max_length": 70,
        "max_gen_length": 140,
        "training_epochs": 10,
        "warmup_steps": 1000,
        "logdir": "logs/Compound_GPT3_zinc20",
        "log_interval": 20,
        "save_interval": 2000,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_save_path": "model_save_path_zinc20",
        "checkpoint_path": None,
        "attention_visualization": False,
        "generate": True,
        "generate_num": 2000
    }

    main()
