# Codebase Readme
This is the codebase for paper: "Revisiting Unsupervised Domain Adaptation Models: a Smoothness Perspective"

## Environment
```
conda env create -f leco.yaml
conda activate leco
```

## Prepare the datasets
Office-31 can be found [here](https://paperswithcode.com/dataset/office-31).  
Office-Home can be found [here](https://www.hemanthdv.org/officeHomeDataset.html).   
Visda-C can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public).     
DomainNet can be found [here](http://ai.bu.edu/M3SDA/).

## Training guides

### Visda-C
For MCC:
```
python da_visda.py --dset visda --lr 0.001 --net resnet101 --gpu_id 0 --batch_size 36 --base MCC --method Blank --interval 2 --s 0 --t 1
```
For MCC + LECO:
```
python da_visda.py --dset visda --lr 0.001 --net resnet101 --gpu_id 0 --batch_size 36 --base MCC --method LECO --interval 2 --s 0 --t 1 --warm_up 3000 --lamda 3
```
We set seed=[2020, 2021, 2022], showing the stable improvements to MCC. Logs can refer to [TV](./log/uda/visda/TV/).
| Methods        | plane | bcycl | bus | car | horse | knife | mcycl | person | plant | sktbrd | train | truck | Per-class |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MCC(2020) | 94.3 | 80.35| 75.93 |64.03 |92.45 |97.16 |85.23 |83.12 |89.23 |86.01 |82.11 |53.26 | 81.93 |
| MCC(2021) | 93.69 |84.06 |76.35 |65.71 |91.39 |94.94 |86.04 |77.62 |92.44 |89.57 |81.52 |54.29 | 82.30 |
| MCC(2022)  | 93.25 |81.18 |73.73 |57.23 |90.94 |71.08 |83.09 |77.05 |82.63 |86.94 |81.89 |55.73 | 77.90 |
| MCC+LeCo(2020) |97.12 |85.96 |83.86 |89.66 |96.55 |97.45 |89.06| 84.05 |95.91 |90.79 |85.08 |43.82| 86.61
| MCC+LeCo(2021) |95.72 |86.33 |86.46 |91.55 |96.18 |96.82 |92.53| 74.18 |96.07 |92.85 |84.07 |38.09| 85.90
| MCC+LeCo(2022) |96.49 |87.02 |79.17 |90.46 |95.86 |96.43 |91.24 |82.55 |94.55 |92.42 |88.36 |40.57| 86.26
For CDAN:
```
python da_visda.py --dset visda --lr 0.01 --net resnet101 --gpu_id 0 --batch_size 36 --base CDAN --method Blank --interval 2 --s 0 --t 1 --warm_up 3000 --lamda 3 --lr_decay2 0.1
```
For CDAN + LECO:
```
python da_visda.py --dset visda --lr 0.01 --net resnet101 --gpu_id 0 --batch_size 36 --base CDAN --method LECO --interval 2 --s 0 --t 1 --warm_up 3000 --lamda 0.5 --lr_decay2 0.1
```
For BNM:
```
python da_visda.py --dset visda --lr 0.001 --net resnet101 --gpu_id 0 --batch_size 36 --base BNM --method Blank --interval 2 --s 0 --t 1
```
FOr BNM + LeCo:
```
python da_visda.py --dset visda --lr 0.001 --net resnet101 --gpu_id 0 --batch_size 36 --base BNM --method LECO --interval 2 --s 0 --t 1 --warm_up 3000 --lamda 2
```

### Office-home
For MCC:
```
python da_home.py --dset office-home --lr 0.01 --net resnet50 --gpu_id 0 --batch_size 36 --base MCC --method Blank --interval 2
```
For MCC + LECO:
```
python da_home.py --dset office-home --lr 0.01 --net resnet50 --gpu_id 0 --batch_size 36 --base MCC --method LECO --interval 2 --warm_up 3000 --lamda 2
```
For CDAN:
```
python da_visda.py --dset visda --lr 0.01 --net resnet101 --gpu_id 0 --batch_size 36 --base CDAN --method Blank --interval 2 --lr_decay2 0.1
```
For CDAN + LECO:
```
python da_visda.py --dset visda --lr 0.01 --net resnet101 --gpu_id 0 --batch_size 36 --base CDAN --method LECO --interval 2 --warm_up 3000 --lamda 2 --lr_decay2 0.1
```
For BNM
```
python da_visda.py --dset visda --lr 0.01 --net resnet101 --gpu_id 0 --batch_size 36 --base BNM --method Blank --interval 2
```
For BNM + LECO
```
python da_visda.py --dset visda --lr 0.01 --net resnet101 --gpu_id 0 --batch_size 36 --base BNM --method LECO --interval 2 --lambda 3
```


### DomainNet
For MCC + LECO
```
python da_domainNet.py --dset com-dn --lr 0.01 --net resnet101 --gpu_id 0 --batch_size 36 --base MCC --method LECO --interval 5 --warm_up 3000 --lamda 2
```
### Office-31
This code file is borrowed from [BNM](https://github.com/cuishuhao/BNM). And you need to specify the source and target domain like follows:  
For baseline: MCC, and method: LECO
```
python da_office.py --gpu_id 0 --base MCC --method LECO --num_iterations 8004  --dset office --s dslr --t amazon --test_interval 2000  --lambda_method 3
```

## Visualization
Intra-class variance and inter-class variance visualization can refer to files ([cal_cluster_intra.py](./cal_cluster_intra.py), [cal_cluster_inter.py](./cal_cluster_inter.py)).

## Validation
Choosing the best hyper-parameter can refer to file: [dev_loss.py](./dev_loss.py).

## BibTeX
```
@inproceedings{wang2022revisiting,
  title={Revisiting Unsupervised Domain Adaptation Models: a Smoothness Perspective},
  author={Xiaodong Wang, Junbao Zhuo, Mengru Zhang, Shuhui Wang, and Yuejian Fang},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  year={2022}
}
```
