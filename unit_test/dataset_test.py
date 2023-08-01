import datasets
import unittest
from functools import partial
from torch.utils.data.dataloader import DataLoader
from transformers import HfArgumentParser
from dataset_gen.preprocess import nuplan_collate_func
from dataclasses import dataclass, field

@dataclass
class TestArguments:
    dataset: str = field(
        default="/home/shiduozhang/nuplan/online_debug/boston_index_demo"
    )
    datadic_path: str = field(
        default="/home/shiduozhang/nuplan/online_debug/"
    )
    batch_size: int = field(
        default=8
    )
    num_workers: int = field(
        default=1
    )
class TestOnlinePreprocess(unittest.TestCase):
    """
    This unittest is designed for online rasterize and data augmentation.
    test cases include:
    1. online rasterize for batch data from dataloader.
    2. test for each augmentation function
    """
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        parser = HfArgumentParser((TestArguments))
        self.args = parser.parse_args()
    
    def test_raster_visulization(self):
        dataset = datasets.load_from_disk(self.args.dataset)
        example = dataset[0]
        example_filename = example['file_name']
        print(example_filename)
        file_index = dataset['file_name'].index(example_filename)
        high_res_raster = example["high_res_raster"].detach().cpu().numpy()
        low_res_raster = example["low_res_raster"].detach().cpu().numpy()
        context_action = example["context_actions"].detach().cpu().numpy()
        trajectory = example["trajectory_label"].detach().cpu().numpy()
        # visulize_raster_perchannel("visulization/rasters/pittsburgh_scenario_test/high", high_res_raster)
        # visulize_raster_perchannel("visulization/rasters/pittsburgh_scenario_test/low", low_res_raster)
        # visulize_trajectory("visulization/rasters/pittsburgh_scenario_test/low",context_action)
        # visulize_trajectory("visulization/rasters/pittsburgh_scenario_test/high",context_action, scale=4)
        print("done")

if __name__ == "__main__":
    unittest.main()