import pandas as pd
from rich import print
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef
import numpy as np
import joblib
import json
import os
# Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


class Model():

    def __init__(self, model, training_path=None, testing_path=None, model_path=None) -> None:
        self.model = model
        self.train_smiles, self.train_labels, self.test_smiles, self.test_labels = self.read_data(
            training_path, testing_path)
        self.train_ecfp = [self.smiles_to_ecfp(s) for s in self.train_smiles]
        self.test_ecfp = [self.smiles_to_ecfp(s) for s in self.test_smiles]
        self.model_path = model_path
        self.train()
        self.prediction = self.predict()

        try:
            self.probabilities = self.predict_proba()
        except:
            print('This model does not support predict_proba()')
            self.probabilities = None

    def train(self):
        self.model.fit(self.train_ecfp, self.train_labels)

    def predict(self):
        return self.model.predict(self.test_ecfp)

    def predict_proba(self):
        return self.model.predict_proba(self.test_ecfp)

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_and_predict_proba(self):
        self.model = joblib.load(self.model_path)
        return self.model.predict_proba(self.test_ecfp)

    def evaluate(self):
        tn, fp, fn, tp = confusion_matrix(
            self.test_labels, self.prediction).ravel()
        sn = tp / (tp + fn) if (tp + fn) != 0 else 0
        sp = tn / (tn + fp) if (tn + fp) != 0 else 0
        acc = accuracy_score(self.test_labels, self.prediction)
        mcc = matthews_corrcoef(self.test_labels, self.prediction)
        return sn, sp, acc, mcc

    def smiles_to_ecfp(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
        return np.array(ecfp)

    def read_data(self, training_path, testing_path):
        df_train = pd.read_csv(training_path)

        # testing_path沒有的話切分training data
        if testing_path == "None":
            df_train, df_test = train_test_split(df_train, test_size=0.2)
        else:
            df_test = pd.read_csv(testing_path)

        X_train = df_train['smiles']
        labels = df_train['labels']

        X_test = df_test['smiles']
        test_labels = df_test['labels']

        return X_train, labels, X_test, test_labels


if __name__ == '__main__':

    with open('./config.json', 'r') as f:
        config = json.load(f)
    # config
    training_path = config['train_data_path']
    testing_path = config['test_data_path']
    kinase = config['kinase_name']  # name of kinase

    if not os.path.exists('./ML_logs'):
        os.mkdir('./ML_logs')

    Random_Forest_model = Model(
        RandomForestClassifier(), training_path, testing_path, kinase)
    Random_Forest_model.save_model(
        './ML_logs/RF_{}.joblib'.format(kinase))

    # 之後補上其他的 model
    sn, sp, acc, mcc = Random_Forest_model.evaluate()
    Random_Forest_prediction = Random_Forest_model.prediction
    print("Model Evaluation:")
    print(f"Sensitivity (Sn): {sn:.3f}")
    print(f"Specificity (Sp): {sp:.3f}")
    print(f"Accuracy (Acc): {acc:.3f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.3f}")
