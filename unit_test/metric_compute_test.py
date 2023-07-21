import evaluate
import torch
import unittest
import torch.nn as nn
from transformers import HfArgumentParser
from transformer4planning.utils import ModelArguments
from transformer4planning.models.model import build_models
from dataclasses import dataclass, field

@dataclass
class TestArguments:
    metric: str = field(
        default="accuracy"
    )

class TestMetricCompute(unittest.TestCase):
    """
    This testcase is designed for hugging face evaluate function
    """
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        parser = HfArgumentParser((TestArguments))
        self.args = parser.parse_args()
    
    def test_metric_compute(self):
        """
        The test result shows that after metric computaion, 
        metric references and predictions before will be cleaned
        """
        metric = evaluate.load(self.args.metric)
        pred = torch.tensor([0, 1, 2])
        gt = torch.tensor([1, 1, 1])
        metric.add_batch(references=gt, predictions=pred)
        first_accuracy = metric.compute() # expect 0.33, 
        metric.add_batch(references=torch.zeros(1), predictions=torch.zeros(1))
        second_accuracy = metric.compute() # expect 1
        print(f"First {self.args.metric}:{first_accuracy}\n, Second {self.args.metric}:{second_accuracy}")

if __name__ == "__main__":
    unittest.main()