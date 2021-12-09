# Covid xRay Binary Classification 
### Using Densenet121 and SIIM-FISABIO-RSNA COVID-19 Detection Database
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-v3.6.13-blue.svg?logo=python&style=for-the-badge" /></a>
<a href="https://www.anaconda.org/"><img src="https://img.shields.io/badge/conda-v4.10.3-blue.svg?logo=conda&style=for-the-badge" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.10.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="https://www.pytorchlightning.ai"><img src="https://img.shields.io/badge/Lightning-v1.3.8-purple.svg?logo=PyTorch-Lightning&style=for-the-badge" /></a>

Negative             |  Typical
:-------------------------:|:-------------------------:
![1e6f48393e17_03](https://user-images.githubusercontent.com/93069949/144041416-d3e5d620-a5b4-45ae-ac8f-331358622b00.png) | ![09cf9767a7bf](https://user-images.githubusercontent.com/93069949/144042579-0e26ae6c-d7c0-439a-b59a-497faf80bdd9.jpg)

## Installation
1. Clone Github
```
import git
git.Git("/your/directory/to/clone").clone("git:https://github.com/ihamdi/Covid-xRay-Classification.git)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; or [download](https://github.com/ihamdi/Covid-xRay-Classification/archive/refs/heads/main.zip) and extract a copy of the files.

2. Create conda environment
```
conda create --name env-name python=3.6.13
```
&nbsp;&nbsp;&nbsp;&nbsp; _*Python 3.6.13 is needed since GDCM is not supported on versions above 3.6._

3. Install [PyTorch](https://pytorch.org/get-started/locally/) according to your machine. For example:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

4. Install dependencies from [`requirements.txt`](https://github.com/ihamdi/Dogs-vs-Cats-Classification/blob/main/requirements.txt) file:
```
pip install -r requirements.txt
```

5. Download Data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Run `python scripts/download_data.py` to download the data using the Kaggle API and extract it automatically. If you haven't used Kaggle API before, please take a look at the instructions at the bottom on how to get your API key.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Otherwise, extract the contents of the "train" directory from the official [SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection/data) page to the [`train`](https://github.com/ihamdi/Covid-xRay-Classification/tree/main/data/train) folder inside the [`data`](https://github.com/ihamdi/Covid-xRay-Classification/tree/main/data/) directory.

## How to use:
### Experiments:
The [`experiment`](https://github.com/ihamdi/Covid-xRay-Classification/tree/main/configs/experiment/) folder inside [`configs`](https://github.com/ihamdi/Covid-xRay-Classification/tree/main/configs/) directory contains a template for configuring an experiment. The easiest way is to make a copy of [`template.yaml`](https://github.com/ihamdi/Covid-xRay-Classification/blob/main/configs/experiment/template.yaml) and edit the parameters accordingly.

If num_classes is set to 4, then the data will be a random mix from all labels. Otherwise, the code will default to binary classification and the data will be 50% negative labeled images and 50% of non-negative labeled images (randomly chosen from the other 3 labels). The program also rejects any patient folders with more than 1 xrays to avoid training on lateral xrays.

To run the default experiment, run the following command
```
python train.py
```
or 
```
python train.py experiment=template
```
This will run an experiment based on the template using the following parameters:
1. 20 epochs (unless early stopping is triggered)
2. Torchxrayvision's "ALL" (pretrained Densenet121) model with no Dropout
3. Adam optimizer with learning rate of 0.003 and AMSGrad enabled.
4. Batch size of 32
5. Number of workers of 10
6. 640 images only
7. 60 : 20 : 20 split
8. Image size of 128x128
9. MIN/(MAX-MIN)x255 normalization
10. No augmentations

### Hyperparameter Search with Optuna:
As part of the Hydra template, Optuna can be used to find the best hyperparameters within a defined range. A template configuration file can be found within [`hparams_search`](https://github.com/ihamdi/Covid-xRay-Classification/tree/main/configs/hparams_search/) folder inside the [`configs`](https://github.com/ihamdi/Covid-xRay-Classification/tree/main/configs/) directory. The template hyperparameter search can be initiated using
```
python run.py -m hparams_search=template_optuna experiment=template
```
or
```
python run.py -m hparams_search=template_optuna experiment=template hydra.sweeper.n_trials=30
```
---
### Background:

Initially, this code was based on my [Dogs vs Cats](https://github.com/ihamdi/Dogs-vs-Cats-Classification) code. I eventually adapted the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) to make it easier to use and log. No submission was made to the Kaggle competition and only the training data is used.

---

### Contact:

For any questions or feedback, please feel free to post comments or contact me at ibraheem.hamdi@mbzuai.ac.ae

---

### Using Kaggle's API:

![image](https://user-images.githubusercontent.com/93069949/144188576-d457568e-7cd2-42f2-ba08-9c41143d674d.png)

![image](https://user-images.githubusercontent.com/93069949/144188635-705e1e29-92ae-4aba-be66-0e1d2e1c29ca.png)

![image](https://user-images.githubusercontent.com/93069949/144188696-f535f9c8-3ed8-4e1b-8f0d-179d7e5be2a2.png)

---

### Referernces:

[Densenet paper](https://arxiv.org/abs/1608.06993) by Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.

[Torchxravision's](https://github.com/mlmed/torchxrayvision) pretrained models for chest xrays.

[Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template/) on Github.
