# Compound generator: A GPT-based drug generator applied within a specific domain.

Workflow

![Image](/image/workflow.png "Workflow")
## Directory Structure
```bash
├── docker-compose.yml
├── Dockerfile
├── image
│   └── workflow.png
├── README.md
├── requirements.txt
└── src
    ├── config.json
    ├── dataset
    │   ├── Inhibitor
    │   │   ├── testing.csv
    │   │   └── training.csv
    │   └── sglt2
    │       └── sglt2.csv
    ├── logs
    ├── machine_model.py
    ├── main.py
    ├── ML_logs
    ├── model_save_path_reward
    ├── model_save_path_zinc20
    │   └── checkpoint-4468000
    ├── output
    │   └── reward_epoch
    ├── __pycache__
    │   └── trainer.cpython-311.pyc
    ├── reward.py
    ├── trainer.py
    ├── visualize.ipynb
    └── zinc20M_gpt2_tokenizer

17 directories, 36 files
```
## Requirements
CompoundGPT currently supports Python > 3.10

- [joblib](https://pypi.python.org/pypi/joblib)
- [NumPy](https://numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [rdkit](https://www.rdkit.org/)
- [pytorch](https://pytorch.org/) > 2.0.0

## Installation
### pip or conda
To install PyTorch, visit the [PyTorch official website](https://pytorch.org) and follow the instructions to download and install the PyTorch version that corresponds to your CUDA version.

Requirements can be installed using pip or conda as
```bash
pip install -r requirements.txt
```
or
```bash
conda list -e > requirements.txt
```

### Docker
If you want to install CompoundGPT using a Docker, we have provided an image for your use.

First check if nvidia-docker is installed for GPU support, if not, please visit the [Nvidia website](https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-vscode-server-step-01.html)

Build image
```bash
docker build -t compound:py310-torch211-cuda121 .
```
execute container
```bash
docker compose up -d
```
enter container 
```bash
docker exec -it compound bash
```
## Getting Started
### Preparing Your Dataset

Format the Data:
Prepare your dataset in the specified CSV format and place it under the /src/dataset/ directory. The expected format for the CSV should be as follows:

```bash
smiles, label
```
For an example of the file format, refer to the sample provided at [here](src/dataset/sglt2/sglt2.csv):

### Configuration
Update Config File:
Once your dataset is prepared and placed in the correct directory, navigate to the [config.json](/src/config.json) file. Update the path in the config file to point to your newly processed dataset.
```bash
"train_data_path": "./dataset/your_path"
"kinase_name": "Name"
```

### Training the Expert System
```bash
python machine_model.py
```
after finish training expert system it will show you performance.
```bash
Model Evaluation:
Sensitivity (Sn): 1.000
Specificity (Sp): 0.997
Accuracy (Acc): 0.999
Matthews Correlation Coefficient (MCC): 0.997
```
### Finetune LLM
After completing the initial training of your model, the next step is to fine-tune your LLM. 

Execute the following command to start the fine-tuning process.
```bash
CUDA_VISIBLE_DEVICES=1 python reward.py
```
During the training process, the model checkpoints for each epoch are saved in a specific directory. Here is how the saving mechanism is set up:

- [Model Checkpoints](/src/model_save_path_reward) : /src/model_save_path_reward
- [Training Outputs](/src/output/reward_epoch) : /src/output/reward_epoch

You can generate or check the differences for each epoch through these two locations.