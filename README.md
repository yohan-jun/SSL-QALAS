# SSL-QALAS: Self-Supervised Learning for Rapid Multiparameter Estimation in Quantitative MRI Using 3D-QALAS

![Alt text](figure/SSL-QALAS.jpg?raw=true "SSL-QALAS")


This is the official code for **"SSL-QALAS: Self-Supervised Learning for Rapid Multiparameter Estimation in Quantitative MRI Using 3D-QALAS"**.

The related paper is published at [Magnetic Resonance in Medicine](https://doi.org/10.1002/mrm.29786).

The baseline code is based on fastMRI code, which is forked from [here](https://github.com/facebookresearch/fastMRI).

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

Note: some of the variables (e.g., turbo factor or echo spacing) might need to be updated in the  `fastmri/models/qalas_map.py` (L287-L288) based on your sequence.

## Training and Validation Logs
To track the training and validation logs, run the tensorboard as below:

```bash
tensorboard --logdir=qalas_log/lightning_logs
```

## Inference
To infer the model, run `inference_qalas_map.py` as below:

```bash
python inference_qalas_map.py --data_path matlab/h5_data/multicoil_val --state_dict_file qalas_log/checkpoints/epoch=XXX-step=XXXX.ckpt --output_path matlab/h5_data
```

The reconstructed maps under `matlab/h5_data/reconstructions` can be read on Matlab using `h5read` matlab function:

```bash
T1 = h5read('train_data.h5','/reconstruction_t1');
T2 = h5read('train_data.h5','/reconstruction_t2');
PD = h5read('train_data.h5','/reconstruction_pd');
IE = h5read('train_data.h5','/reconstruction_ie');
```

## Generating Training and Validation Data
To make .h5 file, run `ssl_qalas_save_h5_from_dicom.m` matlab file

If the same subject data is used for validation (i.e., subject specific training and validation), copy `train_data.h5` and paste under `matlab/h5_data/multicoil_val`.

(Optional) To compare the SSL-QALAS with the reference maps (e.g., dictionary matching results), please put them under `matlab/map_data` (format: .mat file which may contain T1_map, T2_map, PD_map, IE_map, and B1_map)

Sample data can be found [here](https://www.dropbox.com/scl/fo/0lqsttrqavmfxgq32ptkd/h?rlkey=z6f2cnt3243b7us0izac79zj6&dl=0)

## Cite
If you have any questions/comments/suggestions, please contact at yjun@mgh.harvard.edu

If you use the SSL-QALAS code in your project, please cite the following paper:

```BibTeX
@article{jun2023SSL-QALAS,
  title={{SSL-QALAS}: Self-Supervised Learning for rapid multiparameter estimation in quantitative {MRI} using {3D-QALAS}},
  author={Jun, Yohan and Cho, Jaejin and Wang, Xiaoqing and Gee, Michael and Grant, P. Ellen and Bilgic, Berkin and Gagoski, Borjan},
  journal={Magnetic resonance in medicine},
  volume={90},
  number={5},
  pages={2019--2032},
  year={2023},
  publisher={Wiley Online Library}
}
```