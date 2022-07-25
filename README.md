# To Deform or Not: Treatment-aware Longitudinal Registration for Breast DCE-MRI during Neoadjuvant Chemotherapy via Unsupervised Landmarks Detection
Radiologists compare breast DCE-MRI after neoadjuvant chemotherapy (NAC) with pre-treatment scans to evaluate the response to NAC.
Clinical evidence supports that accurate longitudinal deformable registration is key to quantifying tumor changes.
We propose a conditional pyramid registration network based on unsupervised landmark detection and selective volume-preserving to quantify changes over time. 
We use a clinical dataset with 314 patients treated with NAC, yielding a total of 1630 MRI scans.
The results demonstrate that our method registers with better performance than state-of-the-art methods, and achieves better volume-preservation of the tumors at the same time.
Furthermore, a local-global-combining biomarker based on the proposed method is also shown to achieve high accuracy in pathological complete response (pCR) prediction which potentially could be used to avoid unnecessary surgeries for certain patients.
It may be valuable for clinicians and/or computer systems to conduct follow-up tumor segmentation and response prediction on images registered by our method.

## Dataset
The directory structure for our dataset is defined as follows,
```
dataset_root/
|--- list_path/
|    |--- reg/
|         |--- neo_reg_train.csv
|         |--- neo_reg_valid.csv
|         |--- neo_reg_test.csv
|
|--- img_path/
     |--- Patient1/
     |    |--- Scan1/
     |    |    |--- img_T1_1.nii.gz     # T1w image
     |    |    |--- img_sinwas.nii.gz   # Wash-in image
     |    |    |--- seg_breast.nii.gz   # Label of breast segmentation mask
     |    |    |--- seg_tumors.nii.gz   # Label of tumor segmentation mask
     |    |    |--- keypoints.nii.gz    # Heatmap of annotated keypoints
     |    |
     |    |--- Scan2/
     |    |    |--- ...
     |    |
     |    |--- ...
     |     
     |--- Patient2/
     |    |--- ...
     |
     |--- ...
```

Example of `dataset_root/list_path/reg/neo_reg_train.csv`
```csv
aid,mov,fix
Patient1,Scan1,Scan2
Patient1,Scan1,Scan3
Patient1,Scan2,Scan3
Patient2,Scan1,Scan2
...
```

## Usage
### How to train
- Train CPRN as baseline model.
    ```sh
    python src/train_reg_cprn.py -d cuda -s ckpt/cprn
    ```
- Extract structural landmarks.
    ```sh
    python src/train_keynet_S.py -d cuda -s ckpt/kn-s -v vis/kn-s
    ```
- Extract abnormal landmarks and load pretrained model from CPRN.
    ```sh
    python src/train_keynet_A.py -d cuda -s ckpt/kn-a -v vis/kn-a -p ckpt/cprn/ckpt_xxx.pth
    ```
- Train CPRN with structural landmark loss.
    ```sh
    python src/train_reg_cprn_sl.py \
        -d cuda \
        -s ckpt/cprn+sl \
        -c ckpt/cprn/ckpt_xxx.pth \
        -ks ckpt/ks-s/ckpt_xxx.pth
    ```
- Train CPRN with structural landmark loss and abnormal-landmark-based volume-preserving loss.
    ```sh
    python src/train_reg_cprn_sl_vpal.py \
        -d cuda \
        -s ckpt/cprn+sl+vpal \
        -c ckpt/cprn+sl/ckpt_xxx.pth \
        -ks ckpt/ks-s/ckpt_xxx.pth \
        -ka ckpt/ks-a/ckpt_xxx.pth \
        -cprn ckpt/cprn/ckpt_xxx.pth
    ```

## Citation
If this work is helpful for you, please cite our paper as follows:
```bib
TBD
```