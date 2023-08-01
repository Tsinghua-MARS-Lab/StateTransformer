import tensorflow as tf
import torch
import numpy as np
from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops 
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2
from runner import ModelArguments, DataTrainingArguments, HfArgumentParser, build_models
import datasets
from torch.utils.data import DataLoader
import collections
from tqdm import tqdm

def _default_metrics_config():
  config = motion_metrics_pb2.MotionMetricsConfig()
  config_text = """
  track_steps_per_second: 10
  prediction_steps_per_second: 2
  track_history_samples: 10
  track_future_samples: 80
  speed_lower_bound: 1.4
  speed_upper_bound: 11.0
  speed_scale_lower: 0.5
  speed_scale_upper: 1.0
  step_configurations {
    measurement_step: 5
    lateral_miss_threshold: 1.0
    longitudinal_miss_threshold: 2.0
  }
  step_configurations {
    measurement_step: 9
    lateral_miss_threshold: 1.8
    longitudinal_miss_threshold: 3.6
  }
  step_configurations {
    measurement_step: 15
    lateral_miss_threshold: 3.0
    longitudinal_miss_threshold: 6.0
  }
  max_predictions: 6
  """
  text_format.Parse(config_text, config)
  return config

class MotionMetrics():
  """Wrapper for motion metrics computation."""

  def __init__(self, config):
    super().__init__()
    self._prediction_trajectory = []
    self._prediction_score = []
    self._ground_truth_trajectory = []
    self._ground_truth_is_valid = []
    self._prediction_ground_truth_indices = []
    self._prediction_ground_truth_indices_mask = []
    self._object_type = []
    self._metrics_config = config

  def reset_state(self):
    self._prediction_trajectory = []
    self._prediction_score = []
    self._ground_truth_trajectory = []
    self._ground_truth_is_valid = []
    self._prediction_ground_truth_indices = []
    self._prediction_ground_truth_indices_mask = []
    self._object_type = []

  def update_state(self, prediction_trajectory, prediction_score,
                   ground_truth_trajectory, ground_truth_is_valid,
                   prediction_ground_truth_indices,
                   prediction_ground_truth_indices_mask, object_type):
    self._prediction_trajectory.append(prediction_trajectory)
    self._prediction_score.append(prediction_score)
    self._ground_truth_trajectory.append(ground_truth_trajectory)
    self._ground_truth_is_valid.append(ground_truth_is_valid)
    self._prediction_ground_truth_indices.append(
        prediction_ground_truth_indices)
    self._prediction_ground_truth_indices_mask.append(
        prediction_ground_truth_indices_mask)
    self._object_type.append(object_type)

  def result(self):
    # [batch_size, num_preds, 1, 1, steps, 2].
    # The ones indicate top_k = 1, num_agents_per_joint_prediction = 1.
    prediction_trajectory = tf.concat(self._prediction_trajectory, 0)
    # [batch_size, num_preds, 1].
    prediction_score = tf.concat(self._prediction_score, 0)
    # [batch_size, num_agents, gt_steps, 7].
    ground_truth_trajectory = tf.concat(self._ground_truth_trajectory, 0)
    # [batch_size, num_agents, gt_steps].
    ground_truth_is_valid = tf.concat(self._ground_truth_is_valid, 0)
    # [batch_size, num_preds, 1].
    prediction_ground_truth_indices = tf.concat(
        self._prediction_ground_truth_indices, 0)
    # [batch_size, num_preds, 1].
    prediction_ground_truth_indices_mask = tf.concat(
        self._prediction_ground_truth_indices_mask, 0)
    # [batch_size, num_agents].
    object_type = tf.cast(tf.concat(self._object_type, 0), tf.int64)

    # We are predicting more steps than needed by the eval code. Subsample.
    interval = (
        self._metrics_config.track_steps_per_second //
        self._metrics_config.prediction_steps_per_second)
    prediction_trajectory = prediction_trajectory[...,
                                                  (interval - 1)::interval, :]

    return py_metrics_ops.motion_metrics(
        config=self._metrics_config.SerializeToString(),
        prediction_trajectory=prediction_trajectory,
        prediction_score=prediction_score,
        ground_truth_trajectory=ground_truth_trajectory,
        ground_truth_is_valid=ground_truth_is_valid,
        prediction_ground_truth_indices=prediction_ground_truth_indices,
        prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
        object_type=object_type)

def predict():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    model_args.model_name = "scratch-mmtransformer"
    model_args.task = "waymo"
    data_args.saved_valid_dataset_folder = "/home/shiduozhang/waymo/t4p_waymo"
    model = build_models(model_args)
    dataset = datasets.load_from_disk(data_args.saved_valid_dataset_folder)
    dataset.set_format(type='torch')
    dataset.shuffle()
    def waymo_collate_fn(batch):
        from torch.utils.data._utils.collate import default_collate
        expect_keys = expect_keys = ["high_res_raster", "low_res_raster", "context_actions", "trajectory_label"]
        
        elem = batch[0]
        if isinstance(elem, collections.abc.Mapping):
            return {key: default_collate([d[key] for d in batch]) for key in expect_keys}
    test_dataloader = DataLoader(
                dataset=dataset,
                batch_size=2,
                num_workers=1,
                collate_fn=waymo_collate_fn,
                pin_memory=True,
                drop_last=True
            )
    # metric 
    config = _default_metrics_config()
    metric = MotionMetrics(config)
    agent_nums = 128
    for itr, input in enumerate(tqdm(test_dataloader)):
        output = model(**input)
        trajectory_label = input["trajectory_label"][:, :, :2].unsqueeze(1)
        bsz = trajectory_label.shape[0] 
        trajectory_label = torch.cat([trajectory_label, -1 * torch.ones((bsz))])
        # concat to (x, y, width, height, yaw, v_x, v_y)
        trajectory_label = torch.cat([trajectory_label, torch.zeros((bsz, 1, 80, 5)).to(trajectory_label.device)], dim=-1)

        valid_mask = torch.where(trajectory_label[..., 0]!=-1, torch.ones_like(trajectory_label[..., 0]), torch.zeros_like((trajectory_label[..., 0])))
        pred_trajectory = output.logits
        scores = output.scores

        metric.update_state(
            tf.convert_to_tensor(pred_trajectory.reshape(bsz, 6, 1, 1, 80, 2).detach().cpu().numpy()), 
            tf.convert_to_tensor(scores.unsqueeze(-1).detach().cpu().numpy()),
            tf.convert_to_tensor(trajectory_label.detach().cpu().numpy()),
            tf.convert_to_tensor(valid_mask.detach().cpu().numpy().astype(np.bool8)),
            tf.convert_to_tensor(np.ones((bsz, 6, 1), dtype=np.int64)),
            tf.convert_to_tensor(np.ones((bsz, 6, 1), dtype=np.bool8)),
            tf.convert_to_tensor(5 * np.ones((bsz, agent_nums)))
        )
        result = metric.result()

if __name__ == "__main__":
   predict()
        
     