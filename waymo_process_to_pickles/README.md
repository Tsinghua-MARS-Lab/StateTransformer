## To setup environment for pre-processing:

You need to turn waymo tfexample files into pickles first.
To do this, you need to install waymo-open-dataset.

1. create a conda environment

Note waymo-open-dataset does not support any python version above 3.10

`conda env create --name waymo python=3.10`

2. activate the environment
3. install tensorflow

`pip install tensorflow`

For users in China Mainland. use tuna.tsinghua.edu.cn as source and set a higher timeout.

`pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=1000`


4. install waymo-open-dataset

Note you have to install the right version of waymo-open-dataset for your tensorflow version.

`pip install waymo-open-dataset-tf-2-3-0==1.2.0`

## To generate pickles

Run:
`python dataprocess.py {SROURCE_PATH} {TARGET_PATH}`

To generate dataset from pickles:
`python waymo_generation.py --cache_folder /public/MARS/datasets/waymo_motion/waymo_open_dataset_motion_v_1_0_0/cache --num_proc 100`    