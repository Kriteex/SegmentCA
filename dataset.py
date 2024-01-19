import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        #self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform
        self.transform = transform
            
            
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = [file for file in sorted(os.listdir(images_dir)) if file.endswith('.npy')]
        self.masks = [file for file in sorted(os.listdir(masks_dir)) if file.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        
        image = np.load(img_path).transpose(2,0,1)[:3,...]
        mask = Image.open(mask_path)
        mask = np.expand_dims(np.array(mask), axis=0)
        #mask = np.where(mask > 0, 1, 0)
        
        image = self.transform(torch.from_numpy(image).float())
        mask = self.transform(torch.from_numpy(mask).float())
        
        # Create an alpha channel filled with ones, of shape [1, 30, 30]
        alpha_channel = torch.randn(1, 30, 30)

        # Concatenate the alpha channel to the original image along the channel dimension
        # This will change the shape to [4, 30, 30]
        image_with_alpha = torch.cat((image, alpha_channel), 0)

        return image_with_alpha, mask