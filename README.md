# pytorch-lightning-wadb-code-backbone
This repository provides the base code for pytorch-lightning and weight and biases simultaneous integration + hydra (to keep configs clean).
**** Toy configuration of CV classificator

**pytorch-lightning-wadb-code-backbone organization** 

```

│   README.md
│   method.py
|   dataloader.py
|   train.py
|
└───models
│   │   model.py
|
└───datasets
│   │   dataset.py
│   │   transformfactory.py
|
└───configs
│   │   defaults.yaml
│   └─── dataloader
│   │    │  dataset.yaml
│   │
│   └─── model
│   │    │  model.yaml
```

### Code structure ###
The code is divided into a number of subpackages:
- models
- datasets
- configs

### How do I use this code ###
The core of this repository is that the pytorch-lightning (pl) pipline is configured though .yaml file.
There are few key points of this repository:
- write your data preprocessing pipline in dataset file (see the **toy** dataset.py and transformfactory.py)
- write your model and pl logic in model file (see the **toy** model.py)
- configure your pipline through .yaml file and see all metrics in [wadb](https://docs.wandb.ai/)

### Configs ###
To understand the structure see [hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/).
dataset.yaml and model.yaml consist of dataset_type and model_type keys respectively. Through keys values pl pipline is configured.

**Use case:**
Write your dataset pipline (includes preprocessing through transformfactory.py). Pass dataset_type name (as a key) dataset class (as a value) into self.dataset_types dict in dataloader.py file. 

Write your model pipline (includes with train step logic, see **toy** example). Pass model_type name (as a key) model class (as a value) into self.model_types dict in method.py file. 

Done. 

Configure all parameters through .yaml file with integrated [wadb](https://docs.wandb.ai/)
