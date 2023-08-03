import torch
import os
import datasets
import unittest
import pickle
from torch.utils.data.dataloader import DataLoader
from transformers import HfArgumentParser
from transformer4planning.models.model import build_models
from dataset_gen.preprocess import nuplan_collate_func
from runner import ModelArguments, DataTrainingArguments
from functools import partial

class TestGPTfunction(unittest.TestCase):
    """
    This unittest is designed for model base function test and debug.
    test cases include:
        1. forward function test
        2. generate function test
    Just run this file to test above functions. 
    
    """
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
        self.args = parser.parse_args()
        self.args.datadic_path = "/media/shiduozhang/My Passport/nuplan/online_dataset"
        self.args.ar_future_interval=20
        self.args.token_scenario_tag = True
        self.model = build_models(self.args)
        # index dataset load
        index_dataset = datasets.load_from_disk(os.path.join(self.args.datadic_path, "index", "train","train-index_boston"))
        self.dataset = index_dataset.add_column('split', column=['train']*len(index_dataset))
        example = self.dataset[0]
        # map and agent info load
        all_maps_dic = {}
        all_pickles_dic = {}
        map_folder = os.path.join(self.args.datadic_path, 'map')
        for each_map in os.listdir(map_folder):
            if each_map.endswith('.pkl'):
                map_path = os.path.join(map_folder, each_map)
                with open(map_path, 'rb') as f:
                    map_dic = pickle.load(f)
                map_name = each_map.split('.')[0]
                all_maps_dic[map_name] = map_dic
        
        collate_func = partial(nuplan_collate_func, 
                               dic_path=self.args.datadic_path, 
                               autoregressive=self.args.autoregressive, 
                               all_maps_dic=all_maps_dic,
                               all_pickles_dic=all_pickles_dic,)
        
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=2,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_func,
        )

    def test_forward_function(self):
        for _, samples in enumerate(self.dataloader):
            outputs = self.model(**samples)
            generated = self.model.generate(**samples)
            print("Forward function test passed!")
            break
    
if __name__ == "__main__":
    unittest.main()
