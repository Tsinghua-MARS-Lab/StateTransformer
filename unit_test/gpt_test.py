import torch
import os
import datasets
import unittest
import pickle
from torch.utils.data.dataloader import DataLoader
from transformers import HfArgumentParser
from transformer4planning.utils import ModelArguments
from transformer4planning.models.model import build_models
from transformer4planning.preprocess.nuplan_rasterize import nuplan_rasterize_collate_func
from transformer4planning.preprocess.pdm_vectorize import nuplan_vector_collate_func
from runner import  DataTrainingArguments
from functools import partial
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

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
        self.args.model_name = "scratch-gpt-large"
        self.args.datadic_path = "/media/shiduozhang/My Passport/nuplan/online_dataset"
        # self.args.task = "diffusion_decoder"
        # self.args.encoder_type = "vector"
        # index dataset load
        index_dataset = datasets.load_from_disk(os.path.join(self.args.datadic_path, "index", "train","train-index_boston"))
        self.dataset = index_dataset.add_column('split', column=['train']*len(index_dataset))
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
        
        # ----- Rasterize -----#
        self.nuplan_rasterize_collate_func = partial(
                               nuplan_rasterize_collate_func, 
                               dic_path=self.args.datadic_path, 
                               autoregressive=self.args.autoregressive, 
                               all_maps_dic=all_maps_dic,
                               all_pickles_dic=all_pickles_dic,)
        # ----- PDM -----#
        map_api = dict()
        for map in ['sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood']:
            map_api[map] = get_maps_api(map_root="/home/shiduozhang/nuplan/dataset/maps",
                                map_version="nuplan-maps-v1.0",
                                map_name=map
                                )
        self.nuplan_vectorize_collate_func = partial(
                                nuplan_vector_collate_func,
                                dic_path=self.args.datadic_path,
                                map_api=map_api)
        # ----- Diffusion -----#          
        from torch.utils.data._utils.collate import default_collate
        def feat_collate_func(batch, predict_yaw):
            excepted_keys = ['label', 'hidden_state']
            keys = batch[0].keys()
            result = dict()
            for key in keys:
                list_of_dvalues = []
                for d in batch:
                    if key in excepted_keys:
                        d[key] = torch.tensor(d[key])
                        if key == "label" and not predict_yaw:
                            d[key] = d[key][:, :2]
                        list_of_dvalues.append(d[key])
                result[key] = default_collate(list_of_dvalues)
            return result
        
        self.diffusion_collate_func = partial(feat_collate_func, predict_yaw=self.args.predict_yaw)

    def test_nuplan_rasterize(self):
        self.args.predict_yaw = True
        self.args.ar_future_interval=20
        self.args.token_scenario_tag = False
        # build rasterize encoder + mlp decoder model
        self.model = build_models(self.args)
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=self.nuplan_rasterize_collate_func,
        )
        for _, samples in enumerate(self.dataloader):
            outputs = self.model(**samples)
            self.model.eval()
            generated = self.model.generate(**samples)
            print("Nuplan rasterize encoder + mlp forward function test passed!")
            break
    
    def test_pretrained_diffusion_decoder(self):
        """
        dataset is also nuplan rasterize dataset
        """
        self.args.task = "nuplan"
        self.args.predict_yaw = True
        self.args.ar_future_interval=20
        self.args.token_scenario_tag = False
        self.args.key_points_diffusion_decoder_load_from = "/home/shiduozhang/nuplan/checkpoint/NewVer_DiffusionKPTrajDecoder_statedict_for_GPTL_SKPY_loss1_K1_data1_SgFix.pth"
        self.model = build_models(self.args) # change the decoder to pretrained diffusion decoder
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=self.nuplan_rasterize_collate_func,
        )
        for _, samples in enumerate(self.dataloader):
            output = self.model(**samples)
            self.model.eval()
            generated = self.model.generate(**samples)
            print("Nuplan rasterize encoder + diffusion forward function test passed")
            break

    def test_train_diffusion_decoder_only(self):
        """
        Test the model with only diffusion decoder.
        Dataset is changed to feature dataset and the model is just the diffusion decoder to train the decoder only.
        """
        self.args.task = "diffusion_decoder"
        self.dataset = datasets.load_from_disk("/home/shiduozhang/nuplan/diffusion_feat_demo/train")
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=self.diffusion_collate_func,
        )
        self.model = build_models(self.args) # change the decoder to pretrained diffusion decoder
        for _, samples in enumerate(self.dataloader):
            output = self.model(**samples)
            self.model.eval()
            print("Nuplan rasterize encoder + diffusion forward function test passed")
            break

if __name__ == "__main__":
    unittest.main()
