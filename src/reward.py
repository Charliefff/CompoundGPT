
import joblib
import json
import os
from trainer import CustomTrainer
from rdkit.Chem import AllChem
import warnings
import torch
import numpy as np
import pandas as pd
from rich import print
from tqdm import trange
from rdkit import Chem, RDLogger
from dataclasses import dataclass, field
RDLogger.DisableLog('rdApp.error')
warnings.filterwarnings('ignore')

@dataclass
class Reward:
    config: dict
    num: int = field(init=False)
    target: list = field(init=False)
    cutoff: float = field(init=False)
    model: CustomTrainer = field(init=False)
    checkpoint_path: str = field(init=False)
    output: list = field(default_factory=list)
    ML_models: dict = field(default_factory=dict)
    total_training_epochs: int = field(init=False)
    cleaned_smiles: list = field(default_factory=list)
    
    def __post_init__(self):
        self.checkpoint_path = self.config['checkpoint_path']
        self.model = CustomTrainer(['C'], self.config)
        self.num = self.config['total_generate_num']
        self.target = target(self.config['train_data_path'])
        self.total_training_epochs = self.config['total_training_epochs']
        self.cutoff = self.config['cutoff']
        assert self.cutoff < 1
        
        
        # get file path
        filenames = os.listdir(self.config['ML_checkpoint_files'])
        model_files = {name.split('_')[0].strip(): name for name in filenames}
        for model_name, filename in model_files.items():
            if model_name in ['RF', 'MLP', 'SVM', 'LR']:
                self.ML_models[model_name] = joblib.load(os.path.join(self.config['ML_checkpoint_files'], filename))
                
            else:
                print(f"Model {model_name} not found")

    def generate_data(self):

        self.output, self.cleaned_smiles = [], []
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
        # ensemble learning
        for num, model in enumerate(self.ML_models):
            if num == 0:
                probability = self.ML_models[model].predict_proba(ecfp_smiles)
            else:
                probability += self.ML_models[model].predict_proba(ecfp_smiles)
        probability = probability / len(self.ML_models)
        # next training data
        prob1_df = pd.DataFrame([prob[1]
                                for prob in probability], columns=['prob1'])
        smiles_df = pd.DataFrame(self.cleaned_smiles, columns=['SMILES'])
        combined_df = pd.concat([smiles_df, prob1_df], axis=1)
        combined_df = combined_df.sort_values(by='prob1', ascending=False)
        high_prob_df = combined_df[combined_df['prob1'] > 0.75] # high probability data add for next training
        # if high prob data < 10% ,add 10% of the data
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

    def trainer(self):

        epoch, max_mean, stop_sign = 0, 0, 0
        while True:
            print("Epoch     : ", epoch)
            self.generate_data()
            finetune_data, prob_mean = self.calculate_reward(epoch)

            # stop sign
            if max_mean > prob_mean: # no enhancement
                stop_sign += 1
            else: # enhancement
                max_mean = prob_mean
            if len(finetune_data) > int(self.num * self.cutoff): # stop sign 90% of the data is high probability
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
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
    return np.array(ecfp)
        
def main():
    built_file()
    with open('./config.json', 'r') as f:
        config = json.load(f)
    config_model = {
        **config,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "generate_num": config.get('generate_num_per_epoch', 2000)
    }
    reward = Reward(config_model)
    reward.trainer()

if __name__ == '__main__':
    main()
    