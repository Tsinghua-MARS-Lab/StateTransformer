import torch
import torch.nn as nn
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    SE2Index,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.pdm_feature_builder import (
    PDMFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.features.pdm_feature import (
    PDMFeature,
)

# from transformer4planning.models.backbone.str_base import build_model_from_path


class PDMRefOffsetModel(TorchModuleWrapper):
    """
    Wrapper around PDM-Offset MLP that consumes the ego history (position, velocity, acceleration),
    the trajectory of PDM-Closed and the centerline to regresses correction deltas.
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        history_sampling: TrajectorySampling,
        planner: AbstractPlanner,
        centerline_samples: int = 120,
        centerline_interval: float = 1.0,
        hidden_dim: int = 512,
    ):
        """
        Constructor for PDMOffset
        :param trajectory_sampling: Sampling parameters of future trajectory
        :param history_sampling: Sampling parameters of past ego states
        :param planner: Planner for centerline extraction
        :param centerline_samples: Number of poses on the centerline, defaults to 120
        :param centerline_interval: Distance between centerline poses [m], defaults to 1.0
        :param hidden_dim: Size of the hidden dimensionality of the MLP, defaults to 512
        """

        feature_builders = [
            PDMFeatureBuilder(
                trajectory_sampling,
                history_sampling,
                planner,
                centerline_samples,
                centerline_interval,
            )
        ]

        target_builders = [
            EgoTrajectoryTargetBuilder(trajectory_sampling),
        ]

        self.trajectory_sampling = trajectory_sampling
        self.history_sampling = history_sampling

        self.centerline_samples = centerline_samples
        self.centerline_interval = centerline_interval

        self.hidden_dim = hidden_dim

        print('testing model...', self.hidden_dim)

        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=trajectory_sampling,
        )
        


        # self.state_encoding = nn.Sequential(
        #     nn.Linear(
        #         (history_sampling.num_poses + 1) * 3 * len(SE2Index), self.hidden_dim
        #     ),
        #     nn.ReLU(),
        # )
        #

        #
        # self.planner_head = nn.Sequential(
        #     nn.Linear(self.hidden_dim * 3, self.hidden_dim),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     # nn.ReLU(),
        #     # nn.Linear(self.hidden_dim, trajectory_sampling.num_poses * len(SE2Index)),
        # )
        self._model = self.build_model()
        self.loaded_from_checkpoint = False
        
        # self.trajectory_embd = nn.Sequential(
        #     nn.Linear(trajectory_sampling.num_poses*len(SE2Index), self.d_model),
        #     nn.ReLU(),
        # )
        self.centerline_encoding = nn.Sequential(
            nn.Linear(self.centerline_samples * len(SE2Index), int(self.d_model/2)),
            nn.ReLU(),
        )
        #
        self.trajectory_encoding = nn.Sequential(
            nn.Linear(trajectory_sampling.num_poses * len(SE2Index), int(self.d_model/2)),
            nn.ReLU(),
        )

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "pdm_features": PDFeature,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """

        input: PDMFeature = features["pdm_features"]
        # input is a PDMFeature object, can't be directly used as input to the model
        kwargs = {
            "ego_position": input.ego_position,
            "ego_velocity": input.ego_velocity,
            "ego_acceleration": input.ego_acceleration,
            "planner_centerline": input.planner_centerline,
            "planner_trajectory": input.planner_trajectory,
            "high_res_raster": input.high_res_raster,
            "low_res_raster": input.low_res_raster,
            "context_actions": input.context_actions,
            "ego_pose": input.ego_pose,
            }

        batch_size = input.ego_position.shape[0]

        ego_position = input.ego_position.reshape(batch_size, -1).float()
        ego_velocity = input.ego_velocity.reshape(batch_size, -1).float()
        ego_acceleration = input.ego_acceleration.reshape(batch_size, -1).float()

        # encode ego history states
        # state_features = torch.cat(
        #     [ego_position, ego_velocity, ego_acceleration], dim=-1
        # )
        # state_encodings = self.state_encoding(state_features)
        #
        # # encode PDM-Closed trajectory
        # input.planner_trajectory is a tensor of shape (batch_size, num_poses, len(SE2Index))
        planner_trajectory = input.planner_trajectory.float() # bsz 16 3
        # planner_trajectory_embd = self.trajectory_embd(planner_trajectory.reshape(batch_size, -1)) # (batch_size, hidden_dim)
        # planner_trajectory_embd = planner_trajectory_embd.unsqueeze(1)
        trajectory_encodings = self.trajectory_encoding(planner_trajectory.reshape(batch_size, -1))
        #
        # # encode planner centerline
        planner_centerline = input.planner_centerline.reshape(batch_size, -1).float()
        centerline_encodings = self.centerline_encoding(planner_centerline)
        #
        # # decode future trajectory
        planner_features = torch.cat(
            [centerline_encodings, trajectory_encodings], dim=-1
        )
        planner_features = planner_features.unsqueeze(1)

        input_embeds, info_dict = self._model.encoder(is_training=self.training, **kwargs)
        input_embeds = torch.cat((input_embeds[:,:-80,:], planner_features, input_embeds[:,-80:,:]), dim=1)
        
        transformer_outputs = self._model.embedding_to_hidden(input_embeds)
        transformer_outputs_hidden_state = transformer_outputs['last_hidden_state']
        pred_trajectory = self._model.traj_decoder.generate_trajs(transformer_outputs_hidden_state, info_dict)
        
        # align the shape of pred_trajectory with planner_trajectory
        pred_trajectory = torch.cat((pred_trajectory[:, :, :2], pred_trajectory[:, :, 3:]), dim=2)
        pred_trajectory = pred_trajectory[:, ::5, :]
        # pdm_feature = self.planner_head(planner_features)
        # the shape of planner_trajectory is (batch_size, 48)
        # the shape of pred_trajectory is (batch_size, 80, 4)

        output_trajectory = planner_trajectory + pred_trajectory
        output_trajectory = output_trajectory.reshape(batch_size, -1, len(SE2Index))

        return {"trajectory": Trajectory(data=output_trajectory)}

    def build_model(self):
        from transformer4planning.models.backbone.mixtral import STR_Mixtral, STRMixtralConfig
        from transformer4planning.utils.args import (
            ModelArguments,
        )
        default_model_arg = ModelArguments()
        config_p = STRMixtralConfig()
        config_p.update_by_model_args(default_model_arg)
        ModelCls = STR_Mixtral
        # set default setting (based on 200m)
        config_p.n_layer = 16
        config_p.n_embd = config_p.d_model = 320
        config_p.n_inner = config_p.n_embd * 4
        config_p.n_head = 16
        config_p.num_hidden_layers = config_p.n_layer
        config_p.hidden_size = config_p.n_embd
        config_p.intermediate_size = config_p.n_inner
        config_p.num_attention_heads = config_p.n_head
        config_p.use_key_points = "no"
        self.d_model = config_p.d_model

        model = ModelCls(config_p)
        print('PDM+StateTransformer with Reference Initialized!')
        return model
