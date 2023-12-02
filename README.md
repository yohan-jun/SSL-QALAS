# SSL-QALAS: Self-Supervised Learning for Rapid Multiparameter Estimation in Quantitative MRI Using 3D-QALAS

![Alt text](figure/SSL-QALAS.jpg?raw=true "SSL-QALAS")

This is the official code for **"SSL-QALAS: Self-Supervised Learning for Rapid Multiparameter Estimation in Quantitative MRI Using 3D-QALAS"**.\
The related paper is published at [Magnetic Resonance in Medicine](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.29786).

The baseline code is based on fastMRI code, which is forked from [here](https://github.com/facebookresearch/fastMRI)

## Installation
For dependencies and installation, please follow below:

```bash
conda env create -f environment.yml
conda activate ssl_qalas
pip install -e .
```

## Model Training
To train the model, run `train_qalas.py` as below:

```bash
python train_qalas.py --data_path matlab/h5_data --check_val_every_n_epoch 4
```

## Training and Validation Logs
To track the training and validation logs, run the tensorboard as below:

```bash
tensorboard --logdir=qalas_log/lightning_logs
```

## Generating Training and Validation Data
To make .h5 file, run `ssl_qalas_save_h5_from_dicom.m` matlab file

If the same subject data is used for validation (i.e., subject specific training and validation), copy `train_data.h5` and paste under `matlab/h5_data/multicoil_val`.

(Optional) To compare the SSL-QALAS with the reference maps (e.g., dictionary matching results), please put them under `matlab/map_data` (format: .mat file which may contain T1_map, T2_map, PD_map, IE_map, and B1_map)

## Cite
If you have any questions/comments/suggestions, please contact at yjun@mgh.harvard.edu

If you use the SSL-QALAS code in your project, please cite the following paper:

```BibTeX
@inproceedings{jun2023SSL-QALAS,
    title={{SSL-QALAS}: Self-Supervised Learning for rapid multiparameter estimation in quantitative {MRI} using {3D-QALAS}},
    author={Yohan Jun and Jaejin Cho and Xiaoqing Wang and Michael Gee and P. Ellen Grant and Berkin Bilgic and And Borjan Gagoski},
    journal={Magnetic Resonance in Medicine},
    year={2023}
}
```