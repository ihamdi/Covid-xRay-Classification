# Covid xRay Binary Classification 
### Using Pytorch Lightning, Densenet121, and SIIM-FISABIO-RSNA COVID-19 Detection Database
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

If num_classes is set to 4, then the data will be a random mix from all labels. Otherwise, the code will default to binary classification and the data will be an equal mix of negative and non-negative labeled images (randomly chosen from the other 3 classes). The program also rejects any data folders with more than 1 xray files to avoid training on lateral chest xrays.

To run the default experiment, run the following command
```
python train.py
```
or 
```
python train.py experiment=template
```
This will run an experiment based on the template using the following configuation:
1. 20 epochs (unless early stopping is triggered)
2. Torchxrayvision's "ALL" (pretrained Densenet121) model with no Dropout
3. Adam optimizer with learning rate of 0.003 and AMSGrad enabled.
4. Batch size of 32
5. Number of workers of 10
6. 640 images only
7. 60 : 20 : 20 split
8. Image size of 128x128
9. IMG-MIN/(MAX-MIN)x255 normalization
10. No augmentations

*Torchxrayvision models expect 224 so the code defaults to that automatically if one of them is chosen.

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

## Results
When in binary classification mode, the code is able to produce a model with just over 80% accuracy on the validation data before it starts overfitting: 

![W B Chart 12_14_2021, 9_37_59 AM](https://user-images.githubusercontent.com/93069949/145939574-a89b312a-b3ee-42b5-9482-3313f188d22c.png)


This was done using the "All" model from torchxrayvision, Adam optimizer, and the following hyperparameters:
1. drop_rate (Dropout) = 0
2. lr (Learning Rate) = 0.0003
3. amsgrad (for Adam) = True
4. normal (Normalization Method) = 0 (img-min/(max-min)*255) 
5. rotation = 11.355
6. scaling = 0.2789
7. shear = 1
8. translation = 0.07357
9. horizontal_flip = True
10. vertical_flip = True
11. dataset_size (Sample Size) = 3350 (maximum possible to keep subset balanced)
12. train_val_test_split = [70,20,10]
13. batch_size = 156
14. num_workers = 20


Although the code accepts setting num_classes=4, it is currently unable to achieve a validation accuracy higher than 60-62% regardless of hyperparameters:

![W B Chart 12_14_2021, 9_36_37 AM](https://user-images.githubusercontent.com/93069949/145939394-ab623cd7-087d-4256-82d2-358bc6b16d30.png)

F1 Heatmap & Confusion Matrix below show that it is especially not doing well at classifying xrays with Atypical appearance (label 3): 

![media_images_confusion_matrix_winter-totem-6_423_3c808d26766df4b83257](https://user-images.githubusercontent.com/93069949/145939273-8b8bead8-e501-4061-b9c6-9347d0895efb.png)
![media_images_f1_p_r_heatmap_winter-totem-6_423_c6e8de7712ab2e94fcac](https://user-images.githubusercontent.com/93069949/145939274-8b1370d0-bded-4d09-b9d8-db7bdd458954.png)


I am currently investigating whether it is possible to significantly improve the performance of the model. Dropout and Augmentations seem to have an adverse effect on the accuracy so I might need to proceed with a different architecture altogether. 

---

### Background:

Initially, this code was based on my [Dogs vs Cats](https://github.com/ihamdi/Dogs-vs-Cats-Classification) code. I eventually adapted the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) to make it easier to use and log. No submission was made to the Kaggle competition and only the training data is used.

---

### Contact:

For any questions or feedback, please feel free to post comments or contact me at ibraheem.hamdi@mbzuai.ac.ae

---

### Referernces:

[Densenet paper](https://arxiv.org/abs/1608.06993) by Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.

[Torchxravision](https://github.com/mlmed/torchxrayvision)'s page on Github (used for Densenet121 models pretrained on xrays).

[Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)'s page on Github

[Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template/)'s page on Github.

[Weights & Biases](https://wandb.ai/)'s website.

---

### Using Kaggle's API:

![image](https://user-images.githubusercontent.com/93069949/144188576-d457568e-7cd2-42f2-ba08-9c41143d674d.png)

![image](https://user-images.githubusercontent.com/93069949/144188635-705e1e29-92ae-4aba-be66-0e1d2e1c29ca.png)

![image](https://user-images.githubusercontent.com/93069949/144188696-f535f9c8-3ed8-4e1b-8f0d-179d7e5be2a2.png)

### Using Weights & Biases (wandb):

![Screenshot from 2021-1](https://user-images.githubusercontent.com/93069949/145940749-1ee81bbf-a77a-40c6-985c-880ad3a5c6c5.png)
