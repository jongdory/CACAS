from typing import Sequence
import numpy as np
import nibabel as nib
import os
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import SimpleITK as sitk
import torchio as tio
from einops import rearrange, reduce, repeat


def get_augmentations(phase):
    transforms = []
    transforms.append(tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5,99.5)))

    if phase == 'train':
        transforms.append(tio.RandomFlip(p=0.5, axes=['LR', 'AP', 'IS']))
        transforms.append(tio.RandomAffine(p=0.5, degrees=(-30,30,0,0,0,0)))
        transforms.append(tio.RandomGamma(p=0.5, log_gamma=(-0.3, 0.3)))    

    trfms = tio.Compose(transforms)
    return trfms


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    path: str,
    phase: str,
    fold: int = 0,
    seed: int = 55,
    batch_size: int = 1,
    num_workers: int = 4,
):
    '''Returns: dataloader for the model training'''
    dataset = dataset(path, phase, fold, seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True if phase !="test" else False,
    )
    return dataloader


class CarotidDataset(Dataset):
    def __init__(self, 
                 path: str = "", 
                 phase: str = "train",
                 fold: int = 0,
                 seed: int = 55,):
        
        self.path = path
        self.phase = phase
        self.fold = fold
        self.seed = seed
        self.subs = os.listdir(self.path)
        self.pathlist = self.__getlist__()
        self.augmentations = get_augmentations(phase)
        

    def __len__(self):
        return len(self.pathlist)


    def __getfoldmask__(self, li):
        kf = KFold(n_splits = 5, shuffle = True, random_state = self.seed)
        trains = []
        valids = []

        for tr, val in kf.split(li):
            trains.append(tr)
            valids.append(val)

        if self.phase == 'train':
            return trains[self.fold]
        else:
            return valids[self.fold]


    def __getlist__(self):
        pathlist = []

        phase_path = self.path
        sub_list = os.listdir(phase_path)

        idx_mask = self.__getfoldmask__(sub_list)
        sub_list = np.array(sub_list)[idx_mask]

        for sub_i in sub_list:
            sub_path = os.path.join(phase_path, sub_i)

            if os.path.isdir(sub_path):
                pathlist.append(sub_path)

        return pathlist
    

    def __getitem__(self, idx):
        id_ = idx
        path = self.pathlist[idx]

        subject = tio.Subject(
                Image       = tio.ScalarImage(f"{path}/img.nii.gz"), 
                Mask        = tio.LabelMap(f"{path}/roi.nii.gz"), 
                CenterLine  = tio.LabelMap(f"{path}/centerline.nii.gz"),
                WeightMat   = tio.LabelMap(f"{path}/weightmat2.nii.gz"),
            )
        augmented = self.augmentations(subject)
        
        img, mask = augmented['Image'].data.numpy(), augmented['Mask'].data.numpy()
        img, mask = img.astype(np.float32), mask.astype(np.float32)

        cl, wm = augmented['CenterLine'].data.numpy(), augmented['WeightMat'].data.numpy()
        cl, wm = cl.astype(np.float32), wm.astype(np.float32)

        img  = rearrange( img, 'c b h d -> c d h b')
        mask = rearrange(mask, 'c b h d -> c d h b')
        cl   = rearrange(  cl, 'c b h d -> c d h b')
        wm   = rearrange(  wm, 'c b h d -> c d h b')

        img, mask, cl, wm = self.random_crop(img, mask, cl, wm, (256,112,128))

        mask = self.preprocess_mask_labels(mask)
        wm   = np.stack([wm[None], wm[None]]).squeeze()

        return {
            "Id": id_,
            "image": img,
            "mask": mask,
            "centerline": cl,
            "weightmat": wm,
        }

    def random_crop(self, img, mask, centerline, weightmat, crop_size):
        mz, my, mx = img.shape[1:]
        cz, cy, cx = crop_size
        mz, my, mx = mz-cz, my-cy, mx-cx
        
        if self.phase == 'train':
            sz, sy, sx = np.random.randint(mz), np.random.randint(my), np.random.randint(mx)
        else:
            mz, my, mx = img.shape[1:]
            _, sz, sy, sx = center_of_mass(centerline)
            pz, py, px = cz/2, cy/2, cx/2
            sz, sy, sx = sz-pz, sy-py, sx-px

            sz = 0 if sz < 0 else sz
            sy = 0 if sy < 0 else sy
            sx = 0 if sx < 0 else sx
            sz = mz-cz if sz + cz > mz else sz
            sy = my-cy if sy + cy > my else sy
            sx = mx-cx if sx + cx > mx else sx

            sz, sy, sx = int(sz), int(sy), int(sx)

        img       , mask      =        img[:,sz:sz+cz, sy:sy+cy, sx:sx+cx],      mask[:,sz:sz+cz, sy:sy+cy, sx:sx+cx]
        centerline, weightmat = centerline[:,sz:sz+cz, sy:sy+cy, sx:sx+cx], weightmat[:,sz:sz+cz, sy:sy+cy, sx:sx+cx]

        return img, mask, centerline, weightmat


    def preprocess_mask_labels(self, mask: np.ndarray):

        mask_lu = mask.copy()
        mask_lu[mask_lu == 1] = 0
        mask_lu[mask_lu == 2] = 1

        mask_ow = mask.copy()
        mask_ow[mask_ow == 1] = 1
        mask_ow[mask_ow == 2] = 1

        mask = np.stack([mask_lu, mask_ow]).squeeze()

        return mask

class CarotidTestDataset(Dataset):
    def __init__(self, 
                 path: str = "", 
                 phase: str = "test",
                 fold: int = 0,
                 seed: int = 55,
                 sparse: int = 20,):
        
        self.path = path
        self.subs = os.listdir(self.path)

        self.pathlist = self.__getlist__()
        self.augmentations = get_augmentations("test")
        

    def __len__(self):
        return len(self.pathlist) 


    def __getlist__(self):
        pathlist = []
        sub_list = os.listdir(self.path)

        for sub_i in sub_list:
            sub_path = os.path.join(self.path, sub_i)

            pathlist.append(sub_path)

        return pathlist
    

    def __getitem__(self, idx):
        id_ = idx
        path = self.pathlist[idx]

        subject = tio.Subject(
                Image = tio.ScalarImage(f"{path}/img.nii.gz"), 
            )
        augmented = self.augmentations(subject)
        
        img = augmented['Image'].data.numpy().astype(np.float32)

        img  = rearrange( img, 'c b h d -> c d h b')

        return {
            "Id": id_,
            "image": img,
        }


def center_of_mass(input):
    normalizer = np.sum(input) + 1e-8
    grids = np.ogrid[[slice(0, i) for i in input.shape]]

    results = [np.sum(input * grids[dir].astype(float)) / normalizer
               for dir in range(input.ndim)]

    if np.isscalar(results[0]):
        return tuple(results)

    return [tuple(v) for v in np.array(results).T]