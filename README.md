## DTBS: Dual-Teacher Bi-directional Self-training for Domain Adaptation in Nighttime Semantic Segmentation
** by [Fanding Huang], [Zihao Yao], and [Wenhui Zhou]

:bell: **News:**

* [2023-07-15] We are happy to announce that DAFormer was accepted at **ECAI23**.

## Datasets

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**ACDC:** Please, download rgb_anon_trainvaltest.zip and
gt_trainval.zip from [here](https://acdc.vision.ee.ethz.ch/download) and
extract them to `data/acdc`. 

**Dark Zurich :** Please, download the Dark_Zurich_train_anon.zip
and Dark_Zurich_val_anon.zip from
[here](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/) and extract it
to `data/dark_zurich`.

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```


## Training

A training job can be launched using:

```shell
python run_experiments.py --config configs/DTBS/gta2cs_uda_warm_fdthings_rcs_croppl_a999_DTBS.py
```

More experiments in our paper (e.g. network architecture comparison,
component ablations, ...) are coming soon
