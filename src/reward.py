
import joblib
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from trainer import SMILESDataset, CustomTrainer
from rdkit.Chem import rdMolDescriptors
# from rdkit.Chem.rdMolDescriptors import MorganGenerator
from rdkit.Chem import AllChem
import warnings
import torch
import numpy as np
import pandas as pd
from rich import print
from tqdm import trange
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.error')
warnings.filterwarnings('ignore')


def built_file():
    save_reward_model = 'model_save_path_reward'
    output = 'output/reward_epoch'

    if not os.path.exists(save_reward_model):
        os.makedirs(save_reward_model)

    if not os.path.exists(output):
        os.makedirs(output)

# training data


def target(path):
    df_test = pd.read_csv(path)
    df_test = df_test[df_test['labels'] == 1]
    df_test = df_test['smiles'].tolist()
    return df_test

# check smiles


def check_smiles(Smiles: str):
    try:
        mol = Chem.MolFromSmiles(Smiles)
        if mol is None:
            return False
        else:
            return True
    except:
        return False

# ECFP


def smiles_to_ecfp(smiles):

    mol = Chem.MolFromSmiles(smiles)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return np.array(ecfp)


class Reward:
    def __init__(self, config,
                 reward_model: str,
                 ) -> None:

        self.config = config
        self.checkpoint_path = config['checkpoint_path']
        self.model = CustomTrainer(['C'], config)
        self.num = config['total_generate_num']
        assert self.num % config['generate_num'] == 0

        self.target = target(config['train_data_path'])
        self.reward_model = reward_model
        # self.reward_model_1 = reward_model_1
        # self.reward_model_2 = reward_model_2
        # self.reward_model_3 = reward_model_3
        self.total_training_epochs = config['total_training_epochs']
        self.reward_model_path = config['ML_checkpoint_path']
        self.cutoff = config['cutoff']
        self.output = list()
        self.cleaned_smiles = list()

    def generate_data(self):

        self.output = list()
        self.cleaned_smiles = list()
        self.config['generate'] = True

        # generate data
        for i in trange(int(self.num / self.config['generate_num'])):
            smiles = self.model.load_and_generate(self.checkpoint_path, '<S>')
            self.output.extend(smiles)

        # clean output
        for smiles_add_token in self.output:

            cleaned = smiles_add_token.replace("<S>", "").replace(
                "<L>", "").replace("<|endoftext|>", "")
            check = check_smiles(cleaned)
            if check:
                self.cleaned_smiles.append(cleaned)
            else:
                continue

        self.cleaned_smiles = list(set(self.cleaned_smiles))
        self.config['generate'] = False

    def calculate_reward(self, epoch):
        ecfp_smiles = [smiles_to_ecfp(s)
                       for s in self.cleaned_smiles]  # smiles to ecfp
        self.reward_model = joblib.load(self.reward_model_path)
        probability = self.reward_model.predict_proba(ecfp_smiles)
        prob1_df = pd.DataFrame([prob[1]
                                for prob in probability], columns=['prob1'])
        smiles_df = pd.DataFrame(self.cleaned_smiles, columns=['SMILES'])
        combined_df = pd.concat([smiles_df, prob1_df], axis=1)
        combined_df = combined_df.sort_values(by='prob1', ascending=False)
        high_prob_df = combined_df[combined_df['prob1'] > self.cutoff]
        if len(high_prob_df) < 0.1 * len(combined_df):
            high_prob_df = combined_df.head(int(0.1 * len(combined_df)))
        prob1_mean = prob1_df['prob1'].mean()
        combined_df.to_csv(
            f'./output/reward_epoch/epoch_{epoch}.csv')
        # 要不要加入原始data
        # target_df = pd.DataFrame(self.target, columns=['SMILES'])
        # high_prob_df = pd.concat([high_prob_df, target_df], ignore_index=True)

        return high_prob_df, prob1_mean

    def fine_tuning(self, data: pd.DataFrame, epoch: str):

        path = './model_save_path_reward/epoch{}'.format(
            epoch)
        self.config['checkpoint_path'] = self.checkpoint_path
        self.model = CustomTrainer(data['SMILES'].tolist(), self.config)
        self.model.train(reward_model_path=path)
        self.checkpoint_path = './model_save_path_reward/epoch{}'.format(
            epoch)

    def main(self):

        epoch, max_mean, stop_sign = 0, 0, 0
        while True:
            print("Epoch     : ", epoch)
            self.generate_data()
            finetune_data, prob_mean = self.calculate_reward(epoch)

            # stop sign
            if max_mean > prob_mean:
                stop_sign += 1
            else:
                max_mean = prob_mean
            if len(finetune_data) > int(self.num * 0.9):
                break
            if stop_sign > 3:
                print("No Enhancement")
                break
            if epoch == self.total_training_epochs:
                print("Finish Training")
                break
            self.fine_tuning(finetune_data, epoch)
            epoch += 1

        print("Finish Training")


if __name__ == '__main__':

    built_file()
    with open('./config.json', 'r') as f:
        config = json.load(f)

    config_model = {
        "dataset": config['dataset'],
        "gpt_size": config['gpt_size'],
        "checkpoint_path": config['checkpoint_path'],
        "ML_checkpoint_path": config['ML_checkpoint_path'],
        "train_data_path": config['train_data_path'],
        "tokenizer_name": config['tokenizer_name'],
        "learning_rate": config['learning_rate'],
        "batch_size": config['batch_size'],
        "update_loss": config['update_loss'],
        "max_length": config['max_length'],
        "max_gen_length": config['max_gen_length'],
        "training_epochs": config['training_epochs'],
        "total_training_epochs": config['total_training_epochs'],
        "warmup_steps": config['warmup_steps'],
        "logdir": config['logdir'],
        "log_interval": config['log_interval'],
        "save_interval": config['save_interval'],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_save_path": config['model_save_path'],
        "generate": config['generate'],
        # maximum 2000 for each generation memory issue
        "generate_num": config['generate_num_per_epoch'],
        "total_generate_num": config['total_generate_num'],
        "cutoff": config['cutoff']
    }

    # generate 200000 data each epoch
    reward = Reward(config_model,
                    RandomForestClassifier())
    reward.main()
