from pytorch_lightning import LightningDataModule
from torchvision.transforms import transforms
from skimage.transform import resize
from skimage import exposure
from PIL import Image
import torchxrayvision as xrv
import pydicom as dicom
import torchvision
import numpy as np
import warnings
import skimage
import cv2
import sys

class Covid(LightningDataModule):
    def __init__(
        self, 
        data_dir,
        df, 
        classes = 2, 
        mode='train', 
        normal=1, 
        img_size = 128, 
        model = "Densenet121",
        transformations = {"scale": 0, "shear": 0, "translation":[0, 0], 
                        "horizontal_flip": False, "vertical_flip": False, "rotation" : 0}
        ):
        
        self.data_dir=data_dir
        self.df = df
        self.classes=classes
        self.mode=mode
        self.normal=normal
        self.img_size=img_size
        self.model=model
        self.scale = transformations["scale"]
        self.shear = transformations["shear"]
        self.translation = transformations["translation"]
        self.horizontal_flip = transformations["horizontal_flip"]
        self.vertical_flip = transformations["vertical_flip"]
        self.rotation = transformations["rotation"]


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        dcom_path=self.df.iloc[idx].path
        dc_img = dicom.dcmread(dcom_path)
        img_array = dc_img.pixel_array
        
        if self.normal == 1: #histogram normalization
            img_array = exposure.equalize_hist(img_array)
        elif self.normal == 2: #CLAHE normalization
            img_array = exposure.equalize_adapthist(img_array)
        else:
            img_array = (img_array-np.min(img_array))/(np.max(img_array)-np.min(img_array))*255

        img_array = self.preprocess(img_array)
        img = self.transformations(img_array)
        label = int(self.df.iloc[idx].label)
        if self.classes==2 and label != 0:
            label = 1
            
        return img, label

    def preprocess(self, img_array):
        if self.model in ["Densenet121", "Densenet161", "Densenet169", "Densenet201"]:
            img_array = resize(img_array, (self.img_size, self.img_size), anti_aliasing=True)
            img_array = img_array*255 #PIL.Image.fromarray expects 0-255 integers
            img_array = np.repeat(img_array[:, :, np.newaxis], 3, axis=2) #Densenet expects 3-channel image
            img_array = Image.fromarray(img_array.astype('uint8'))
        elif self.model in ["RSNA", "NIH", "PadChest", "CheXpert", "MIMIC_NB", "MIMIC_CH", "ALL"]:
            img_array = img_array[None, :, :]
            transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                        XRayResizer(224, engine="cv2")])    #torchxrayvision models expect resized images 
            img_array = transform(img_array)
            img_array = np.transpose(img_array, (1, 2, 0)) #torchxrayvision models expect (pixels, pixels, label) instead of (label, pixels, pixels)
            img_array = img_array.astype('uint8')
        return img_array
        
    def transformations(self,img):
        scale=[1-self.scale,1+self.scale]
        translation=[0,self.translation]
        if self.model in ["Densenet121", "Densenet161", "Densenet169", "Densenet201"]:
            if self.mode == 'train':
                if self.horizontal_flip == True and self.vertical_flip == True:
                    self.transformations = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                            transforms.RandomVerticalFlip(),
                                                            transforms.RandomAffine(translate = translation,
                                                                                    degrees=self.rotation, 
                                                                                    scale = scale,
                                                                                    shear=self.shear),
                                                            transforms.ToTensor()]
                                                            )        
                elif self.horizontal_flip == True:
                    self.transformations = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                            transforms.RandomAffine(translate = translation,
                                                                                    degrees=self.rotation, 
                                                                                    scale = scale, 
                                                                                    shear=self.shear),
                                                            transforms.ToTensor()]
                                                            ) 
                elif self.vertical_flip == True:
                    self.transformations = transforms.Compose([transforms.RandomVerticalFlip(),
                                                            transforms.RandomAffine(translate = translation,
                                                                                    degrees=self.rotation, 
                                                                                    scale = scale, 
                                                                                    shear=self.shear),
                                                            transforms.ToTensor()]
                                                            )   
                else:
                    self.transformations = transforms.Compose([transforms.RandomAffine(translate = translation,
                                                                                        degrees=self.rotation, 
                                                                                        scale = scale, 
                                                                                        shear=self.shear),
                                                            transforms.ToTensor()]
                                                            )
            else:
                self.transformations = transforms.Compose([transforms.RandomAffine(translate = translation,
                                                                                    degrees=self.rotation, 
                                                                                    scale = scale, 
                                                                                    shear=self.shear), 
                                                            transforms.ToTensor()]
                                                            )

        elif self.model in ["ALL", "RSNA", "NIH", "PadChest", "CheXpert", "MIMIC_NB", "MIMIC_CH", "JFH"]:
            if self.mode == 'train':
                if self.horizontal_flip == True and self.vertical_flip == True:
                    self.transformations = transforms.Compose([transforms.ToPILImage(), 
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomVerticalFlip(),
                                                            transforms.RandomAffine(translate = translation,
                                                                                    degrees=self.rotation, 
                                                                                    scale = scale, 
                                                                                    shear=self.shear),
                                                            transforms.ToTensor()]
                                                            )        
                elif self.horizontal_flip == True:
                    self.transformations = transforms.Compose([transforms.ToPILImage(), 
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomAffine(translate = translation,
                                                                                    degrees=self.rotation, 
                                                                                    scale = scale, 
                                                                                    shear=self.shear),
                                                            transforms.ToTensor()]
                                                            ) 
                elif self.vertical_flip == True:
                    self.transformations = transforms.Compose([transforms.ToPILImage(),
                                                            transforms.RandomVerticalFlip(),
                                                            transforms.RandomAffine(translate = translation,
                                                                                    degrees=self.rotation, 
                                                                                    scale = scale, 
                                                                                    shear=self.shear),
                                                            transforms.ToTensor()]
                                                            )   
                else:
                    self.transformations = transforms.Compose([transforms.ToPILImage(), 
                                                            transforms.RandomAffine(translate = translation,
                                                                                    degrees=self.rotation, 
                                                                                    scale = scale, 
                                                                                    shear=self.shear),
                                                            transforms.ToTensor()]
                                                            )
            else:
                self.transformations = transforms.Compose([transforms.ToPILImage(), 
                                                            transforms.RandomAffine(translate = translation,
                                                                                    degrees=self.rotation, 
                                                                                    scale = scale, 
                                                                                    shear=self.shear), 
                                                            transforms.ToTensor()]
                                                            )
        return self.transformations(img)

    

class XRayResizer(object): #Modified from torchxrayvision to remove the unnecessary warning about 
                            ##setting XRayResizer engine to cv2 even when engine is set to cv2
    def __init__(self, size, engine="skimage"):
        self.size = size
        self.engine = engine
        if 'cv2' in sys.modules and engine != 'cv2':
            print("Setting XRayResizer engine to cv2 could increase performance.")

    def __call__(self, img):
        if self.engine == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.transform.resize(img, (1, self.size, self.size), 
                                                mode='constant', preserve_range=True).astype(np.float32)
        elif self.engine == "cv2":
            return cv2.resize(img[0,:,:], 
                              (self.size, self.size), 
                              interpolation = cv2.INTER_AREA
                             ).reshape(1,self.size,self.size).astype(np.float32)
        else:
            raise Exception("Unknown engine, Must be skimage (default) or cv2.")