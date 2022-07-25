import torch
from torch.utils.data import Dataset

import os
import numpy as np
import random
import SimpleITK as sitk

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


class DatasetReg(Dataset):
    def __init__(
        self,
        img_path='path/to/img/',
        list_path='path/to/filelist/',
        mode='train',
    ):
        """
        mode: ['train', 'valid', 'test']
        """
        self.mode = mode
        self.image_list = []

        with open(os.path.join(list_path, 'neo_reg_{}.csv'.format(mode)), 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                row = line.replace('\n', '').split(',')
                aid = row[0]
                scan_mov = row[1]
                scan_fix = row[2]
                self.image_list.append({
                    'm': os.path.join(img_path, aid, scan_mov),
                    'f': os.path.join(img_path, aid, scan_fix),
                })
    
    def preprocess(self, x, flip=False):
        if self.mode=='train':
            if flip:
                x = x[:, :, :, ::-1]
        x = x.copy()
        return x
    
    def __getitem__(self, index):
        path = self.image_list[index]
        mov_p = path['m']
        fix_p = path['f']
        
        img_T1_m = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mov_p, 'img_T1_1.nii.gz')))
        sinwas_m = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mov_p, 'img_sinwas.nii.gz')))
        seg_breast_m = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mov_p, 'seg_breast.nii.gz')))
        seg_tumor_m = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mov_p, 'seg_tumors.nii.gz')))

        img_T1_f = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(fix_p, 'img_T1_1.nii.gz')))
        sinwas_f = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(fix_p, 'img_sinwas.nii.gz')))
        seg_breast_f = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(fix_p, 'seg_breast.nii.gz')))
        seg_tumor_f = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(fix_p, 'seg_tumors.nii.gz')))

        img_T1_m = np.clip(img_T1_m, 0, 3000) / 3000.
        img_T1_f = np.clip(img_T1_f, 0, 3000) / 3000.
        sinwas_m = np.clip(sinwas_m, 0, 3000) / 3000.
        sinwas_f = np.clip(sinwas_f, 0, 3000) / 3000.

        img_T1 = np.stack([img_T1_m, sinwas_m, seg_breast_m, seg_tumor_m,
                           img_T1_f, sinwas_f, seg_breast_f, seg_tumor_f], axis=0)

        if self.mode!='train':
            seg_keypoints_m = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mov_p, 'keypoints.nii.gz')))
            seg_keypoints_f = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(fix_p, 'keypoints.nii.gz')))
            img_T1 = np.concatenate([img_T1, np.stack([seg_keypoints_m, seg_keypoints_f], axis=0)], axis=0)

        if self.mode=='train':
            seed = random.randint(0, 1)
        else:
            seed = 0
        img_T1 = self.preprocess(img_T1, flip=seed)

        return {
            'mov': torch.from_numpy(img_T1[0:2,:,:,:]),
            'mov_seg': torch.from_numpy(img_T1[2:3,:,:,:]),
            'mov_tumor': torch.from_numpy(img_T1[3:4,:,:,:]),
            'fix': torch.from_numpy(img_T1[4:6,:,:,:]),
            'fix_seg': torch.from_numpy(img_T1[6:7,:,:,:]),
            'fix_tumor': torch.from_numpy(img_T1[7:8,:,:,:]),
            'mov_keypoints': torch.from_numpy(img_T1[8:9,:,:,:]) if self.mode!='train' else 0,
            'fix_keypoints': torch.from_numpy(img_T1[9:10,:,:,:]) if self.mode!='train' else 0,
            'aid': os.path.basename(os.path.dirname(mov_p)),
            'name_mov': mov_p,
            'name_fix': fix_p,
        }

    def __len__(self):
        return len(self.image_list)