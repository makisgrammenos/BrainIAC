import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd,
    NormalizeIntensityd,
    RandAffined, RandFlipd, RandGaussianNoised, RandGaussianSmoothd,
    RandAdjustContrastd, ToTensord
)

def get_default_transform(image_size=(96,96,96)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandAffined(
            keys=["image"],
            rotate_range=(0.1, 0.1, 0.1),
            translate_range=(5, 5, 5),
            scale_range=(0.1, 0.1, 0.1),
            prob=0.5,
            padding_mode="border"
        ),
        # Only left-right flipping (typically axis 2 for brain MRI in RAS orientation)
        # This preserves anatomical correctness while providing useful augmentation
        RandFlipd(keys=["image"], spatial_axis=[2], prob=0.5),  # Left-right flip only
        #ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        RandGaussianSmoothd(keys=["image"], prob=0.2),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
        RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.3)),
        
        ToTensord(keys=["image"])
    ])

def get_validation_transform(image_size=(96,96,96)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        #ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ToTensord(keys=["image"])
    ])

class ADNIDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None,path_col='nifti_path', label_col='label'):
        self.dataframe = pd.read_csv(csv_path, dtype={"pat_id": str, "dataset": str})
        #drop nan values on label column and print how many samples were dropped
        initial_len = len(self.dataframe)
        self.dataframe = self.dataframe.dropna(subset=[label_col])
        dropped_len = initial_len - len(self.dataframe)
        if dropped_len > 0:
            print(f"Dropped {dropped_len} samples due to NaN values in label column '{label_col}'")
        self.root_dir = root_dir
        self.transform = transform if transform is not None else get_default_transform()
        self.path_col = path_col
        self.label_col = label_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        pat_id = str(self.dataframe.loc[idx, self.path_col])
        label = self.dataframe.loc[idx, self.label_col]  
        #dataset = str(self.dataframe.loc[idx, 'dataset'])
        
        # Construct image path for MCI/Stroke format
        # img_path = os.path.join(self.root_dir,  pat_id  + ".nii.gz")
        img_path = os.path.join(self.root_dir, pat_id)
        sample = {"image": img_path}
        sample = self.transform(sample)
        return {"image": sample["image"], self.label_col: torch.tensor(label, dtype=torch.float32)}

