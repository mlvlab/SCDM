import os
import PIL
from PIL import Image

def resize(target_path, save_path, size, opt):
    for item in os.listdir(target_path):
        img = Image.open(os.path.join(target_path,item))
        img = img.resize(size = (size,size), resample = opt)
        img.save(os.path.join(save_path,item))
    
    

target_path = "data/ADE20K/ADEChallengeData2016/annotations/validation"
save_path = "data/ADE20K_noisy/ADE20K_DS/ADEChallengeData2016/annotations/validation"
size = 64
opt = PIL.Image.NEAREST
# os.makedirs(save_path)

resize(target_path,save_path,size,opt)