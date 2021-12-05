from kaggle.api.kaggle_api_extended import KaggleApi
import os
import zipfile
import shutil
import tqdm

api = KaggleApi()
api.authenticate()

competition = "carvana-image-masking-challenge"
download_path = "data/"

api.competition_download_file(competition,file_name="train_hq.zip",path=download_path)
api.competition_download_file(competition,file_name="train_masks.zip",path=download_path)

images = zipfile.ZipFile(download_path+"train_hq.zip", 'r')
for image in tqdm.tqdm(images.filelist,desc="Extracting images", unit=" image"):
    images.extract(image,path=download_path)
for i in tqdm.tqdm(os.listdir(download_path+"train_hq/"),desc="Moving images to imgs folder",unit=" image"):
    if i not in os.listdir(download_path+"imgs/"):
        shutil.move(download_path+"train_hq/"+i,download_path+"imgs/")

masks = zipfile.ZipFile(download_path+"train_masks.zip", 'r')
for mask in tqdm.tqdm(masks.filelist,desc="Extracting masks", unit=" mask"):
    masks.extract(mask,path=download_path)
for i in tqdm.tqdm(os.listdir(download_path+"train_masks/"),desc = "Moving masks to masks folder", unit=" mask"):
    if i not in os.listdir(download_path+"masks/"):
        shutil.move(download_path+"train_masks/"+i,download_path+"masks/")

print("Cleaning up...")
shutil.rmtree(download_path+"train_hq")
os.remove(download_path+"train_hq.zip")
shutil.rmtree(download_path+"train_masks")
os.remove(download_path+"train_masks.zip")