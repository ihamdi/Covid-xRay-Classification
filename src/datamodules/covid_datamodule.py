from typing import Optional, Tuple
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import glob

from src.datamodules.datasets.covid import Covid


class COVIDDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (60, 20, 20),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,

        csv_name: str = "train_study_level.csv",
        classes: int = 2,
        dataset_size: int = 500,
        normal: int = 1,
        img_size: int = 128,
        model: str = "Densenet121",
        scale: float = 0, 
        shear: float = 0, 
        translation: int = 0,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        rotation: int = 0,

    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.csv_name = csv_name
        self.classes = classes
        self.dataset_size = dataset_size
        self.normal = normal
        self.img_size = img_size
        self.model = model
        self.scale = scale
        self.shear = shear
        self.translation = translation
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation

        self.transformations={"scale": self.scale, 
                            "shear": self.shear, 
                            "translation": self.translation,
                            "horizontal_flip": self.horizontal_flip,
                            "vertical_flip": self.vertical_flip,
                            "rotation": self.rotation
        }
        
        train_df, valid_df, test_df = self.preparing_data(path = self.data_dir+self.csv_name)

        self.data_train: Optional[Dataset] = Covid(df = train_df, 
                                                data_dir = self.data_dir, 
                                                classes = self.classes, 
                                                model = self.model,
                                                mode='train', 
                                                transformations = self.transformations)
        self.data_val: Optional[Dataset] = Covid(df = valid_df,
                                                data_dir = self.data_dir,
                                                classes = self.classes, 
                                                model = self.model,
                                                mode='valid',
                                                normal = self.normal,
                                                img_size = self.img_size,
                                                transformations = self.transformations)
        self.data_test: Optional[Dataset] = Covid(df = test_df, 
                                                data_dir = self.data_dir, 
                                                classes = self.classes, 
                                                model = self.model, 
                                                mode='test', 
                                                normal = self.normal, 
                                                img_size = self.img_size,
                                                transformations = self.transformations)

    def preparing_data(self, path: str):
        oset_fol=path[:-((len(path.split("/")[-1])+1))]
        train_dir = oset_fol+'/train'
        
        df = pd.read_csv(path)
        df = df.rename(columns = {'Negative for Pneumonia': 'Negative', 'Typical Appearance': 'Typical', 'Indeterminate Appearance': 'Indeterminate', 'Atypical Appearance': 'Atypical'}, inplace = False)

        df.loc[df['Negative'] == 1, 'label'] = 0
        df.loc[df['Typical'] == 1, 'label'] = 1
        df.loc[df['Indeterminate'] == 1, 'label'] = 2
        df.loc[df['Atypical'] == 1, 'label'] = 3

        negs_df=df.loc[df['label']==0]
        typ_df=df.loc[df['label']==1]
        inter_df=df.loc[df['label']==2]
        atyp_df=df.loc[df['label']==3]

        if self.classes == 4:
            df = pd.concat([negs_df, typ_df, inter_df, atyp_df]).reset_index(drop=True)
        else:
            if self.dataset_size != 0:
                negs_df = negs_df.sample(int(self.dataset_size/2))
            poss_df=df.loc[df['Negative']==0]
            poss_df = poss_df.sample(len(negs_df))
            df = pd.concat([negs_df, poss_df]).reset_index(drop=True)
        
        if self.dataset_size != 0:
                df = df.sample(int(self.dataset_size))

        removed=0
        for index, row in df.iterrows():
            if len(glob.glob(train_dir+"/"+row.id.split('_')[0]+"/*/"))>1:
                df = df.drop(df.index[df["id"] == row.id.split('_')[0]+"_study"])
                removed+=1
        print("Removed",removed,"patients with more than 1 files to eliminate lateral x-rays")

        df['path'] = df.apply(lambda row: glob.glob(train_dir+"/"+row.id.split('_')[0]+"/*/*.dcm")[0], axis=1)

        train_df, test_df = train_test_split(df, 
                                            test_size=self.train_val_test_split[2], 
                                            random_state=42,
                                            stratify=df.Negative.values
                                            )
        train_df, valid_df = train_test_split(train_df,
                                            test_size=self.train_val_test_split[1]/self.train_val_test_split[0], 
                                            random_state=42,
                                            stratify=train_df.Negative.values
                                            )
        return train_df, valid_df, test_df


    @property
    def num_classes(self) -> int:
        return self.classes

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
