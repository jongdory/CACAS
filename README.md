# Carotid Artery Segmentation using Center point Annotation

#### Overview

![fig1](/asset/fig1.png)



## Dataset

We used CosMos 2022 Challenge Dataset

please download the dataset through this

**Train dataset**: https://zenodo.org/record/6481870#.Y7aqaXZBzdk

**Test dataset**: https://zenodo.org/record/6843257#.Y7aqSHZBzdk



## How to use

1. Download the data set from the link above and preprocess the dataset

2. Extract the center point from the given label

3. Make a learning plan using Totalloss at loss.py.




## Train

```shell
python main.py --mode total --model Unet3D --fold 0 
```



## Citation

If you find this repo useful for your research, please consider citing our paper:

```bibtex
@inproceedings{,
  title={},
  author={},
  booktitle={},
  pages={},
  year={2023},
  organization={}
}
```