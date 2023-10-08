from multiprocessing import Pool
from omegaconf import OmegaConf
from dataset.waymo_dataset_v1_aug import WaymoDatasetV1Aug
from utils.logger import log


def main_map(i):
    dataset_config = OmegaConf.load('configs/config_v1_aug.yaml')
    dataset = WaymoDatasetV1Aug(dataset_config.DATA_CONFIG, True, log)
    index_list = list(range(100*i, 100*(i+1)))
    for index in index_list:
        if index<len(dataset):
            data_dict = dataset[index]
    print('finish ', i)
    return
 
 
if __name__ == '__main__':

    dataset_config = OmegaConf.load('configs/config_v1_aug.yaml')
    dataset = WaymoDatasetV1Aug(dataset_config.DATA_CONFIG, True, log)
    data_len = len(dataset)
 
    seg_num = data_len // 100 + 1
    inputs = list(range(seg_num))
    
    pool = Pool(60)
    pool_outputs = pool.map(main_map, inputs)

# 48702