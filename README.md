# Covid xRay Binary Classification 
### Using Densenet121 and SIIM-FISABIO-RSNA COVID-19 Detection Database Database
<a href="https://www.anaconda.org/"><img src="https://img.shields.io/badge/conda-v4.10.3-blue.svg?logo=conda&style=for-the-badge" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.10.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-v3.6.13-blue.svg?logo=python&style=for-the-badge" /></a>

Negative             |  Typical
:-------------------------:|:-------------------------:
![1e6f48393e17_03](https://user-images.githubusercontent.com/93069949/144041416-d3e5d620-a5b4-45ae-ac8f-331358622b00.png) | ![09cf9767a7bf](https://user-images.githubusercontent.com/93069949/144042579-0e26ae6c-d7c0-439a-b59a-497faf80bdd9.jpg)

## Installation
1. Clone Github
```
import git
git.Git("/your/directory/to/clone").clone("git:https://github.com/ihamdi/Covid-xRay-Classification.git)
```

or [download](https://github.com/ihamdi/Covid-xRay-Classification/archive/refs/heads/main.zip) files.

2. Create conda environment
```
conda create --name xray python=3.6.13
```

3. Install [PyTorch](https://pytorch.org/get-started/locally/)

4. Install dependencies
```
pip install -r requirements.txt
```

5. Download the data

The code is designed to download the data directly using the Kaggle API and extract it automatically. If you haven't used Kaggle API before, please look at the section at the bottom on how to download your API key. 

Otherwise, download the dataset from the official [Dogs vs. Cats Challenge](https://www.kaggle.com/c/dogs-vs-cats/data) page and extract train.zip to where the Jupyter notebook is located on your machine.

## How to use:
The [`configs`](https://github.com/Covid-xRay-Classification/configs/experiment/) directory contains a template for creating different experiments using:

1. Number of epochs
2. Model
3. Optimizer
4. Batch size
5. Number of workers
6. Size of subset used for the experiment
7. Training : Validation: Testing split ratio
8. Normalization methods
9. Augmentations
10. Patience used for early stopping
11. Project and id name for Wandb loggin

To run the default experiment, run the following command
```
python train.py
```

This will run the template experiment which is setup to run with the following parameters:
1. 20 epochs (unless early stopping is triggered)
2. Torchxrayvision's "ALL" (pretrained Densenet121 model) with no Dropout
3. Adam optimizer with learning rate of 0.003 and AMSGrad enabled.
4. Batch size of 32
5. Number of workers of 10
6. 640 images only
7. 60 : 20 : 20 split
8. Image size of 128x128
9. MIN/(MAX-MIN)x255 normalization
10. No augmentations


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
[Torchxravision's](https://github.com/mlmed/torchxrayvision) pretrained Densenet models for chest xrays.
