import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from ldm.util import instantiate_from_config
from omegaconf import DictConfig, ListConfig
import os
from einops import rearrange

def make_dataset():
    transform = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c')),
        ]
    )
    ##file path for image directory
    return WoolDataSet('/home/ubuntu/images', transform)
    
class WoolDataSet(Dataset):
    def __init__(self,img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
       
    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idx):
        data = {}
        filename = os.listdir(self.img_dir)[idx]
        caption = os.path.splitext(filename)[0]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        data["image"] = image
        data["caption"] = caption
        return data        