import datasets
import unittest
import os
from functools import partial
from torch.utils.data.dataloader import DataLoader
from transformers import HfArgumentParser
from dataset_gen.preprocess import nuplan_collate_func
from dataclasses import dataclass, field
from transformer4planning.checkratser import visulize_raster_perchannel, visulize_trajectory

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
    raster_output_path: str = field(
        default="visulization/rasters/online_test/"
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

    def test_augmentation(self):
        pass

    def test_rasterize_inbatch(self):
        """
        Load data from dataloader to check if the collate_fn works well
        """
        dataset = datasets.load_from_disk(self.args.dataset)
        collate_fn = partial(nuplan_collate_func, dic_path=self.args.datadic_path, autoregressive=True)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            # collate_fn=index_collate_fn,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True        
        )
        for _, example_batch in enumerate(dataloader):
            print(example_batch["high_res_raster"].shape)
            print(example_batch["low_res_raster"].shape)
            break

    def test_visulize_raster(self):
        """
        Visulize rasters to check if the online process is correct
        """
        dataset = datasets.load_from_disk(self.args.dataset)
        collate_fn = partial(nuplan_collate_func, dic_path=self.args.datadic_path, autoregressive=True)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            # collate_fn=index_collate_fn,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True        
        )
        for _, example_batch in enumerate(dataloader):
            visulize_raster_perchannel(os.path.join(self.args.raster_output_path, "high"), example_batch["high_res_raster"][0].detach().cpu().numpy())
            visulize_raster_perchannel(os.path.join(self.args.raster_output_path, "low"), example_batch["low_res_raster"][0].detach().cpu().numpy())
            print(example_batch["trajectory"][0].detach().cpu().numpy())
            break

if __name__ == "__main__":
    unittest.main()
    