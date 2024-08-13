import copy
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
from transformer4planning.models.decoder.base import TrajectoryDecoder
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")


@dataclass
class LTMOutput(CausalLMOutputWithCrossAttentions):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    pred_dict: Optional[Dict[str, torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_items: Optional[Dict[str, torch.FloatTensor]] = None


def fetch_ith_sample(inputs, i):
    sample = {}
    for key in inputs.keys():
        sample[key] = inputs[key][i]
    return sample


def move_np_to_torch(nparray, device, dtype):
    return torch.tensor(nparray, device=device, dtype=dtype)


def list_of_sample_dic_to_batch(list_of_sample_dic):
    batch_size = len(list_of_sample_dic)
    batched_dic = {}
    for each_key in list_of_sample_dic[0]:
        if isinstance(list_of_sample_dic[0][each_key], np.ndarray):
            shape = list_of_sample_dic[0][each_key].shape
            batched_data = np.zeros((batch_size, *shape))
            for i in range(batch_size):
                batched_data[i] = list_of_sample_dic[i][each_key]
            batched_dic[each_key] = batched_data
        elif each_key in ['aug_current']:
            if each_key not in batched_dic:
                batched_dic[each_key] = []
            batched_dic[each_key] = np.array([list_of_sample_dic[i][each_key] for i in range(batch_size)])
    return batched_dic


class STRConfig(PretrainedConfig):
    def update_by_model_args(self, model_args):
        if hasattr(model_args, "__dict__"):
            for each_key in model_args.__dict__:
                self.__dict__[each_key] = model_args.__dict__[each_key]
        else:
            for each_key in model_args:
                self.__dict__[each_key] = model_args[each_key]
        # to be compatible with older models
        attr_list = ["use_key_points", "kp_decoder_type", "separate_kp_encoder", "use_proposal",
                     "autoregressive_proposals", "selected_exponential_past",
                     "rms_norm", "residual_in_fp32", "fused_add_norm", "raster_encoder_type",
                     "vit_intermediate_size", "mean_circular_loss",
                     "camera_image_encoder", "use_speed", "no_yaw_with_stepping", "autoregressive", "regression_long_class_short",
                     "kp_dropout", "traj_dropout", "trajectory_decoder_type", "skip_yaw_norm",
                     "output_router_logits"]
        for each_attr in attr_list:
            if not hasattr(self, each_attr):
                self.__dict__[each_attr] = False


class STR(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.encoder = None
        self.traj_decoder = None
        self.camera_image_encoder = None
        self.k = int(self.config.k)

        self.use_proposal = self.config.use_proposal
        # if self.use_proposal: assert self.config.task == "waymo", "NotImplemented"

        self.use_key_points = self.config.use_key_points
        self.kp_decoder_type = self.config.kp_decoder_type

        self.model_parallel = False
        self.device_map = None
        self.clf_metrics = None

        self.kp_tokenizer = None
        try:
            if self.config.kp_tokenizer is not None:
                self.key_points_decoder = nn.ModuleList()
                self.build_tokenizer()
        except AttributeError:
            pass

        # Initialize weights and apply final processing
        self.build_encoder()
        self.build_decoder()

        self.training_scenarios = None
        if self.config.output_router_logits:
            logger.info("Now using z-loss for MoE router balancing.")

    def build_tokenizer(self):
        # For key points, only use x, and y
        # currently forcing key point to be 2 dimension, with no speed and no yaw
        if self.config.kp_tokenizer == 'uniform':
            from transformer4planning.models.tokenizer.uniform_kp_tokenizer import UniformKPTokenizer
            if self.config.use_key_points == 'specified_backward':
                self.kp_tokenizer = []
                x_min = [-45, -7, -4, -2, -1.5]
                x_max = [180, 86, 45, 25, 15]
                y_min = [-70, -25, -8, -3, -1]
                y_max = [80, 28, 9, 2.5, 1]
                key_point_number = [[50, 50], [20, 20], [10, 10], [5, 5], [3, 3]]
                for i in range(5):
                    kp_tokenizer = UniformKPTokenizer(
                        num_key_points=key_point_number[i],
                        x_min=x_min[i],
                        x_max=x_max[i],
                        y_min=y_min[i],
                        y_max=y_max[i]
                    )
                    self.kp_tokenizer.append(kp_tokenizer)
            elif self.config.use_key_points == 'specified_4s':
                kp_tokenizer = UniformKPTokenizer(
                    num_key_points=[20, 20],
                    x_min=-7,
                    x_max=86,
                    y_min=-25,
                    y_max=28,
                )
                self.kp_tokenizer.append(kp_tokenizer)

        elif self.config.kp_tokenizer == 'cluster':
            from transformer4planning.models.tokenizer.cluster_kp_tokenizer import ClusterKPTokenizer
            cluster_path = self.config.kp_cluster_files.split(",")
            logger.warning(f"cluster_path {cluster_path}")
            if self.config.use_key_points == 'specified_backward':
                assert len(cluster_path) == 5, f"we only support 8s 4s 2s 1s 0_5s csvfiles but get {cluster_path}"
            elif self.config.use_key_points == 'specified_4s':
                assert len(cluster_path) == 1, f"we only support 4scsvfiles but get {cluster_path}"
            self.kp_tokenizer = []
            for i in range(len(cluster_path)):
                kp_tokenizer = ClusterKPTokenizer(cluster_path[i])
                self.kp_tokenizer.append(kp_tokenizer)
        elif self.config.kp_tokenizer == 'cluster_traj':
            from transformer4planning.models.tokenizer.cluster_traj_tokenizer import ClusterTrajTokenizer
            cluster_path = self.config.kp_cluster_files.split(",")
            logger.warning(f"cluster_path {cluster_path}")
            self.kp_tokenizer = []
            for i in range(len(cluster_path)):
                kp_tokenizer = ClusterTrajTokenizer(cluster_path[i])
                self.kp_tokenizer.append(kp_tokenizer)
        else:
            raise NotImplementedError

    def build_encoder(self):
        if self.config.task == "nuplan":
            if "raster" in self.config.encoder_type:
                if self.config.autoregressive:
                    from transformer4planning.models.encoder.nuplan_raster_encoder import NuplanRasterizeAutoRegressiveEncoder
                    self.encoder = NuplanRasterizeAutoRegressiveEncoder(self.config)
                else:
                    from transformer4planning.models.encoder.nuplan_raster_encoder import NuplanRasterizeEncoder
                    self.encoder = NuplanRasterizeEncoder(self.config)

                if self.kp_tokenizer is not None:
                    self.encoder.kp_tokenizer = self.kp_tokenizer
            elif "vector" in self.config.encoder_type:
                from transformer4planning.models.encoder.pdm_encoder import PDMEncoder
                pdm_kwargs = dict(
                    hidden_dim=self.config.n_embd,
                    centerline_dim=120,
                    history_dim=20
                )
                self.encoder = PDMEncoder(pdm_kwargs, self.config)
            else:
                raise AttributeError("encoder_type should be either raster or vector")
        elif self.config.task == "waymo":
            from transformer4planning.models.encoder.waymo_vector_encoder import WaymoVectorizeEncoder
            action_kwargs = dict(
                d_embed=self.config.n_embd
            )
            self.encoder = WaymoVectorizeEncoder(action_kwargs, self.config)
        else:
            raise NotImplementedError

    def build_decoder(self):
        # load pretrained diffusion keypoint decoder
        #TODO: add diffusion decoder trained from scratch
        if self.use_proposal:
            if self.config.task == "nuplan":
                from transformer4planning.models.decoder.base import ProposalDecoderCLS
                self.proposal_decoder = ProposalDecoderCLS(self.config, proposal_num=self.use_proposal)
            elif self.config.task == "waymo":
                from transformer4planning.models.decoder.base import ProposalDecoder
                self.proposal_decoder = ProposalDecoder(self.config)

        if self.use_key_points != 'no':
            if self.kp_decoder_type == "diffusion":
                from transformer4planning.models.decoder.diffusion_decoder import KeyPointDiffusionDecoder
                self.key_points_decoder = KeyPointDiffusionDecoder(self.config)
                if self.config.key_points_diffusion_decoder_load_from is not None:
                    logger.info(f"Now loading pretrained key_points_diffusion_decoder from {self.config.key_points_diffusion_decoder_load_from}.")
                    state_dict = torch.load(self.config.key_points_diffusion_decoder_load_from)
                    self.key_points_decoder.model.load_state_dict(state_dict)
                    logger.info("Pretrained keypoint decoder has been loaded!")
                else:
                    logger.info("Now initializing diffusion decoder from scratch. Training will consume lots of time.")
            elif self.kp_decoder_type == "linear":
                from transformer4planning.models.decoder.base import KeyPointLinearDecoder
                self.key_points_decoder = KeyPointLinearDecoder(self.config)
            elif self.kp_decoder_type == "mlp":
                if self.config.regression_long_class_short:
                    assert self.config.kp_tokenizer == 'cluster', "only support cluster tokenizer for regression_long_class_short"

                if self.kp_tokenizer is None:
                    from transformer4planning.models.decoder.base import KeyPointMLPDeocder
                    self.key_points_decoder = KeyPointMLPDeocder(self.config)
                elif self.config.kp_tokenizer == "uniform":
                    from transformer4planning.models.decoder.base import KeyPointDecoderCLS
                    if self.config.use_key_points == 'specified_backward':
                        proposal_nums = [50*50, 20*20, 10*10, 5*5, 3*3]
                        for i in range(5):
                            new_key_points_decoder = KeyPointDecoderCLS(self.config, proposal_num=proposal_nums[i])
                            new_key_points_decoder.kp_tokenizer = self.kp_tokenizer[i]
                            self.key_points_decoder.append(new_key_points_decoder)
                    elif self.config.use_key_points == 'specified_4s':
                        new_key_points_decoder = KeyPointDecoderCLS(self.config, proposal_num=20*20)
                        new_key_points_decoder.kp_tokenizer = self.kp_tokenizer[0]
                        self.key_points_decoder.append(new_key_points_decoder)
                elif self.config.kp_tokenizer == 'cluster':
                    from transformer4planning.models.decoder.base import KeyPointDecoderCLS
                    if self.config.regression_long_class_short:
                        for i in range(5):
                            if i in [0, 1, 2]:
                                from transformer4planning.models.decoder.base import KeyPointMLPDeocder
                                self.key_points_decoder.append(KeyPointMLPDeocder(self.config))
                    else:
                        if self.config.use_key_points == 'specified_backward':
                            for i in range(5):
                                proposal_num_i = self.kp_tokenizer[i].centers.shape[0]
                                new_key_points_decoder = KeyPointDecoderCLS(self.config, proposal_num=proposal_num_i)
                                new_key_points_decoder.kp_tokenizer = self.kp_tokenizer[i]
                                self.key_points_decoder.append(new_key_points_decoder)
                        elif self.config.use_key_points == 'specified_4s':
                            proposal_num_i = self.kp_tokenizer[0].centers.shape[0]
                            new_key_points_decoder = KeyPointDecoderCLS(self.config, proposal_num=proposal_num_i)
                            new_key_points_decoder.kp_tokenizer = self.kp_tokenizer[0]
                            self.key_points_decoder.append(new_key_points_decoder)

            elif self.kp_decoder_type == "candi_cls":
                from transformer4planning.models.decoder.base import KeyPointDecoderEmbedAndCLS
                if self.config.kp_tokenizer == 'cluster':
                    if self.config.use_key_points == 'specified_4s':
                        proposal_num_i = self.kp_tokenizer[0].centers.shape[0]
                        new_key_points_decoder = KeyPointDecoderEmbedAndCLS(self.config, proposal_num=proposal_num_i)
                        new_key_points_decoder.kp_tokenizer = self.kp_tokenizer[0]
                        self.key_points_decoder.append(new_key_points_decoder)
        if self.use_key_points == 'no':
            if self.config.kp_tokenizer == 'cluster_traj':
                from transformer4planning.models.decoder.base import TrajDecoderCLS
                proposal_num_i = self.kp_tokenizer[0].trajs.shape[0]
                new_traj_decoder = TrajDecoderCLS(self.config, proposal_num=proposal_num_i)
                new_traj_decoder.kp_tokenizer = self.kp_tokenizer[0]
                self.key_traj_decoder = new_traj_decoder
        # create a model list of traj_decoder for each
        # self.traj_decoders = nn.ModuleList()
        # for i in range(80):
        #     self.traj_decoders.append(TrajectoryDecoder(self.config))
        self.traj_decoder = TrajectoryDecoder(self.config)

    def _prepare_attention_mask_for_generation(self, input_embeds):
        return torch.ones(input_embeds.shape[:2], dtype=torch.long, device=input_embeds.device)

    def _prepare_position_ids_for_generation(self, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    def forward(
            self,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_dict:
            raise NotImplementedError('need to return dict for evaluations in trainer.py')

        input_embeds, info_dict = self.encoder(is_training=self.training, **kwargs)
        transformer_outputs = self.embedding_to_hidden(input_embeds, return_dict=return_dict)
        transformer_outputs_hidden_state = transformer_outputs['last_hidden_state']

        if self.config.autoregressive:
            frames_length_to_predict = info_dict["pred_length"]
            trajectory_label = info_dict['trajectory_label'][..., -frames_length_to_predict:, :]
            loss = torch.tensor(0, dtype=input_embeds.dtype, device=transformer_outputs_hidden_state.device)
            traj_loss, traj_logits = self.traj_decoder.compute_traj_loss_autoregressive(transformer_outputs_hidden_state,
                                                                                        trajectory_label,
                                                                                        info_dict)
            loss += traj_loss
        else:
            trajectory_label = info_dict["trajectory_label"]
            if self.config.reverse_traj_index_order:
                trajectory_label = trajectory_label.flip(-2)
            loss = torch.tensor(0, dtype=input_embeds.dtype, device=transformer_outputs_hidden_state.device)
            frames_length_to_predict = trajectory_label.shape[1]
            if self.config.kp_tokenizer == 'cluster_traj':
                traj_loss, traj_logits = self.key_traj_decoder.compute_traj_loss(transformer_outputs_hidden_state, info_dict)
                loss += traj_loss
            else:
                traj_loss, traj_logits = self.traj_decoder.compute_traj_loss(transformer_outputs_hidden_state,
                                                                            trajectory_label,
                                                                            info_dict)
                if not self.config.pred_key_points_only:
                    loss += traj_loss

        loss_items = dict(
            traj_loss=traj_loss,
        )
        pred_dict = {"traj_logits": traj_logits}
        # if finetuning_with_stepping:
        #     logger.warning('passing gt for debugging')
        #     pred_dict = {"traj_logits": trajectory_label}

        if self.use_proposal:
            if self.config.task == "nuplan":
                proposal_loss, pred_proposal_cls = self.proposal_decoder.compute_proposal_loss(transformer_outputs_hidden_state, info_dict)
                loss += proposal_loss
                loss_items["proposal_loss"] = proposal_loss
                pred_dict["proposal"] = pred_proposal_cls
                # debugging
                pred_proposal_score = pred_proposal_cls.softmax(-1)
                topk_score, topk_indx = torch.topk(pred_proposal_score[:, 0, :], dim=-1, k=self.k)

            elif self.config.task == "waymo":
                proposal_loss, proposal_loss_logits = self.proposal_decoder.compute_proposal_loss(transformer_outputs_hidden_state, info_dict)
                loss += proposal_loss
                loss += proposal_loss_logits
                loss_items["proposal_loss"] = proposal_loss
                pred_dict["proposal"] = proposal_loss_logits

        if self.config.dense_pred:
            assert self.config.task == "waymo"
            loss += info_dict["dense_pred_loss"]
            loss_items["dense_pred_loss"] = info_dict["dense_pred_loss"]

        if self.use_key_points != 'no' and not self.config.pred_traj_only:
            loss_per_kp = None
            # new: if pred_traj_only, no need to compute key points, this setting is for putting key point from other pretrained model later
            if self.config.generate_diffusion_dataset_for_key_points_decoder:
                future_key_points = info_dict["future_key_points"] if self.config.predict_yaw else \
                            info_dict["future_key_points"][..., :2]
                self.key_points_decoder.save_features(input_embeds,info_dict["context_length"],info_dict,future_key_points,transformer_outputs_hidden_state)

            if self.config.kp_decoder_type == "diffusion":
                # assert not self.training, "please train diffusion decoder separately."
                # return a dummy loss&kp_logits here. The real data for computing metrics will be computed in the generate function
                kp_loss = torch.tensor(0.0).to(transformer_outputs_hidden_state.device)
                kp_logits = info_dict["future_key_points"][..., :2].to(transformer_outputs_hidden_state.device)
            elif self.config.kp_decoder_type == "linear":
                kp_loss, kp_logits = self.key_points_decoder.compute_keypoint_loss(transformer_outputs_hidden_state, info_dict)
            else:
                if self.kp_tokenizer is None:
                    kp_loss, kp_logits, loss_per_kp = self.key_points_decoder.compute_keypoint_loss(transformer_outputs_hidden_state, info_dict)
                    # kp_loss will be 10x larger than traj_loss when converged
                    if self.k > 1:
                        kp_logits = kp_logits['selected_logits']
                else:
                    assert self.k == 1, "only support k=1 for now"
                    kp_losses, kp_logits = [], []
                    for i in range(len(self.key_points_decoder)):
                        if self.config.regression_long_class_short and i in [0, 1, 2]:
                            kp_loss, kp_logit, loss_per_kp = self.key_points_decoder[i].compute_keypoint_loss(transformer_outputs_hidden_state, info_dict)
                            kp_logits.append(kp_logit[:, i, :].squeeze(1))
                            kp_losses.append(loss_per_kp[i])
                        else:
                            kp_loss, kp_id = self.key_points_decoder[i].compute_keypoint_loss(transformer_outputs_hidden_state, info_dict, i)
                            kp_losses.append(kp_loss)  # list of [1]
                            kp_logit = self.kp_tokenizer[i].decode(kp_id, dtype=kp_loss.dtype, device=kp_loss.device)
                            kp_logits.append(kp_logit)  # list of [bsz, 2]

                    loss_per_kp = torch.stack(kp_losses, dim=0)  # [5]
                    loss_per_kp = loss_per_kp[None, :] # [1, 5]
                    kp_loss = loss_per_kp.mean()
                    kp_logits = torch.stack(kp_logits, dim=1)  # [bsz, 5, 2]
            if self.config.predict_yaw:
                # padding last dimension from 2 to 4
                kp_logits = torch.cat([kp_logits, torch.zeros_like(kp_logits)[:, :, :2]], dim=-1)

            loss += kp_loss
            traj_logits = torch.cat([kp_logits, traj_logits], dim=1)
            pred_dict["kp_logits"] = kp_logits
            loss_items["kp_loss"] = kp_loss
            if loss_per_kp is not None:
                loss_items["loss_per_kp"] = loss_per_kp

        if self.config.output_router_logits:
            aux_loss = self.load_balancing_loss_func(
                transformer_outputs["router_logits"],
                self.config.num_local_experts,
                self.config.num_experts_per_token
            )
            loss += self.config.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        # WIP: training with simulations
        if self.config.finetuning_with_simulation_on_val and self.training:
            sim_loss = self.do_closed_loop_simulation(kwargs)
            loss += sim_loss

        # if not return_dict:
        #     output = (traj_logits,) + transformer_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output
        return LTMOutput(
            loss=loss,
            logits=traj_logits,  # deprecated, use pred_dict for evaluation instead
            pred_dict=pred_dict,
            hidden_states=transformer_outputs_hidden_state,
            loss_items=loss_items
        )

    def embedding_to_hidden(self, input_embeds, attention_mask=None, position_ids=None, return_dict=True):
        # default as the gpt2 model output
        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict
        )
        return transformer_outputs

    def do_closed_loop_simulation(self,
                                  inputs,
                                  metric_key_prefix: str = "CLS", ):
        """
        Run closed loop simulation and returns the metrics.
        Args:
            val1k_datasest (`Dataset`, *optional*):
                Pass a dataset. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method
                 are automatically removed. It must implement the `__len__` method.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "8sFDE" will be named
                "CLS_8sFDE" if the prefix is "CLS" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        from run_simulation import build_simulation_in_batch
        from nuplan_simulation.planner_utils import build_metrics_aggregators
        from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback
        from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
        import datetime, time, os

        experiment_name = 'closed_loop_nonreactive_agents'  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
        job_name = 'STR_planner'
        experiment_time = datetime.datetime.now()
        experiment = os.path.join(experiment_name, job_name, str(experiment_time))
        output_dir = os.path.join(f"tempForTrainSim")
        simulation_dir = "simulation"
        metric_dir = "metrics"
        aggregator_metric_dir = "aggregator_metric"

        # initialize main aggregator
        metric_aggregators = build_metrics_aggregators(experiment_name, output_dir, aggregator_metric_dir)
        metric_save_path = f"{output_dir}/{metric_dir}"
        metric_aggregator_callback = MetricAggregatorCallback(metric_save_path, metric_aggregators)
        main_callbacks = MultiMainCallback([metric_aggregator_callback])
        main_callbacks.on_run_simulation_start()
        # set a timer
        start_time = time.perf_counter()
        scenarios = self.training_scenarios
        # filter scenarios
        scenario_ids_in_batch = inputs['scenario_id']
        filtered_scenarios = [each for each in scenarios if each.token in scenario_ids_in_batch]

        # begin testing
        logger.info('Running simulations at training...')

        if not hasattr(self, 'all_road_dic'):
            self.all_road_dic = {}

        batch_size = len(scenario_ids_in_batch)
        _, overall_score_dic, overall_score = build_simulation_in_batch(experiment_name, filtered_scenarios, output_dir, simulation_dir, metric_dir,
                                                                        batch_size=batch_size,
                                                                        model=self,
                                                                        all_road_dic=self.all_road_dic,
                                                                        save_reports=False)
        main_callbacks.on_run_simulation_end()
        # overall_score_dic['overall_score'] = overall_score
        logger.info('Simulation at training Done')

        # compute loss
        sim_loss = torch.tensor(1.0 / max(0.01, overall_score), device=self.device)

        logger.info(f'\nTime all: {time.perf_counter() - start_time:.3f} s with loss {sim_loss:.3f}')
        return sim_loss


    @torch.no_grad()
    def generate(self, **kwargs) -> torch.FloatTensor:
        # first encode context
        input_embeds, info_dict = self.encoder(is_training=False, **kwargs)
        batch_size, _, _ = input_embeds.shape
        device = input_embeds.device
        context_length = info_dict["context_length"]

        # for debug only
        gt_1s_kp = kwargs.get('gt_1s_kp', None)

        if self.use_proposal:
            if self.config.autoregressive_proposals:
                # TODO: Training for debugging results
                proposal_result = []
                proposal_scores = []
                assert self.config.task == 'nuplan', 'waymo proposal autoregressive not implemented yet'
                dummy_proposal_embedding = self.encoder.proposal_m_embed(torch.zeros((batch_size, int(self.config.proposal_num), 1), device=device))  # bsz, 16, 256
                input_embeds[:, context_length:context_length+int(self.config.proposal_num), :] = dummy_proposal_embedding
                # loop over each intention for generation
                for i in range(int(self.config.proposal_num)):
                    context_embeds = input_embeds[:, :context_length + 1 + i, :]
                    attention_mask = torch.ones(context_embeds.shape[:2], dtype=torch.long, device=device)
                    position_ids = self._prepare_position_ids_for_generation(attention_mask.clone())
                    transformer_outputs_hidden_state = self.embedding_to_hidden(context_embeds)['last_hidden_state']
                    proposal_hidden_state = transformer_outputs_hidden_state[:, context_length - 1 + i:context_length - 1 + 1 + i, :]  # (bs, 1, n_embed)
                    proposal_pred_score = self.proposal_decoder.proposal_cls_decoder(proposal_hidden_state).softmax(-1)  # (bs, 1, 5)
                    # WARNING: Only tested with self.k = 1
                    topk_score, topk_indx = torch.topk(proposal_pred_score[:, 0, :], dim=-1, k=self.k)
                    # topk_score: (bs, 5) topk_indx: (bs, 1)
                    proposal_pred_embed = self.encoder.proposal_m_embed(topk_indx.float())  # (bs, n_embed)
                    # print('test generate 1: ', topk_indx.unsqueeze(-1).float().shape, topk_indx, topk_score, proposal_pred_score[:, 0, :])
                    proposal_result.append(topk_indx.unsqueeze(1))  # list of (bs, 1, 1)
                    proposal_scores.append(proposal_pred_score[:, 0, :].unsqueeze(1))  # list of (bs, 1, 13)
                    input_embeds[:, context_length+i:context_length+i+1, :] = proposal_pred_embed.unsqueeze(1)
                proposal_result = torch.cat(proposal_result, dim=1)  # (bs, 13, 1)
                proposal_scores = torch.cat(proposal_scores, dim=1)  # (bs, 13, 5)
            else:
                if self.config.task == "nuplan":
                    dummy_proposal_embedding = self.encoder.proposal_m_embed(torch.zeros((batch_size, 1), device=device)).unsqueeze(1)
                elif self.config.task == 'waymo':
                    dummy_proposal_embedding = self.encoder.proposal_m_embed(torch.zeros((batch_size, 2), device=device)).unsqueeze(1)
                input_embeds[:, context_length:context_length+1, :] = dummy_proposal_embedding

                context_embeds = input_embeds[:, :context_length+1, :]
                attention_mask = torch.ones(context_embeds.shape[:2], dtype=torch.long, device=device)
                transformer_outputs_hidden_state = self.embedding_to_hidden(
                    input_embeds=context_embeds,
                    attention_mask=attention_mask,
                    position_ids=self._prepare_position_ids_for_generation(attention_mask.clone())
                )['last_hidden_state']
                proposal_hidden_state = transformer_outputs_hidden_state[:, context_length-1:context_length-1+1, :] # (bs, 1, n_embed)
                proposal_pred_score = self.proposal_decoder.proposal_cls_decoder(proposal_hidden_state).softmax(-1) # (bs, 1, 64/5)
                if self.config.task == "nuplan":
                    # WARNING: Only tested with self.k = 1
                    topk_score, topk_indx = torch.topk(proposal_pred_score[:, 0, :], dim=-1, k=self.k)
                    # topk_score: (bs, 5), topk_indx: (bs, 1)
                    proposal_pred_embed = self.encoder.proposal_m_embed(topk_indx.float())  # (bs, n_embed)
                    # print('test generate 1: ', topk_indx.unsqueeze(-1).float().shape, topk_indx, topk_score, proposal_pred_score[:, 0, :])
                    proposal_result = topk_indx
                    proposal_scores = proposal_pred_score[:, 0, :]
                    # proposal_pred_embed: (bs, k, n_embed)
                elif self.config.task == 'waymo':
                    proposal_logit = info_dict["center_obj_proposal_pts"] # (bs, 64, 2)
                    topk_score, topk_indx = torch.topk(proposal_pred_score[:, 0, :], dim=-1, k=self.k)
                    proposal_pred_logit = proposal_logit[torch.arange(batch_size)[:, None].repeat(1, self.k).view(-1), topk_indx.view(-1), :].view(batch_size, self.k, 2)
                    proposal_pred_embed = self.encoder.proposal_m_embed(proposal_pred_logit)
                    proposal_result = topk_indx
                    proposal_scores = proposal_pred_score[:, 0, :]

        traj_logits_k = []
        key_points_logits_k = []
        select_k = 1 if self.config.task == 'nuplan' else self.k

        for mode in range(select_k):
            if self.use_proposal:
                if self.config.autoregressive_proposals:
                    # already updated in previous step
                    pass
                else:
                    if self.config.task == "nuplan":
                        input_embeds[:, context_length:context_length + 1, :] = proposal_pred_embed.unsqueeze(1)
                    elif self.config.task == 'waymo':
                        input_embeds[:, context_length:context_length + 1, :] = proposal_pred_embed.unsqueeze(2)[:, mode, :, :]

            if self.use_key_points != "no":
                pred_length = info_dict["pred_length"]
                selected_indices = self.encoder.selected_indices
                kp_start_index = int(context_length)
                if self.config.kp_decoder_type == "candi_cls":
                    candi_kp_num = info_dict['candi_kp_num']
                    kp_start_index += candi_kp_num

                if self.use_proposal:
                    if self.config.autoregressive_proposals:
                        kp_start_index += int(self.config.proposal_num)
                    else:
                        kp_start_index += 1

                # pass the following infos during generate for one sample (non-batch) generate with KP checking
                map_name = kwargs.get("map", None)
                route_ids = kwargs.get("route_ids", None)
                ego_pose = kwargs.get("ego_pose", None)
                road_dic = kwargs.get("road_dic", None)

                trajectory_label_dummy = torch.zeros((batch_size, pred_length, 4), device=device)
                if 'specified' in self.use_key_points:
                    future_key_points = trajectory_label_dummy[:, selected_indices, :]
                else:
                    ar_future_interval = 20
                    future_key_points = trajectory_label_dummy[:, ar_future_interval - 1::ar_future_interval, :]

                assert future_key_points.shape[1] > 0, 'future points not enough to sample'

                if self.config.task == "nuplan":
                    if not self.config.separate_kp_encoder:
                        assert False, 'deprecated, use separate_kp_encoder instead'
                        if self.config.use_speed:
                            # padding speed, padding the last dimension from 4 to 7
                            future_key_points = torch.cat([future_key_points, torch.zeros_like(future_key_points)[:, :, :3]], dim=-1)
                        future_key_embeds_dummy = self.encoder.action_m_embed(future_key_points)
                    else:
                        future_key_embeds_dummy = self.encoder.kps_m_embed(future_key_points[..., :2])
                else:
                    assert False, 'Key Point for waymo not implemented yet'

                key_points_num = future_key_points.shape[1]

                input_embeds[:, kp_start_index:kp_start_index + key_points_num, :] = future_key_embeds_dummy
                pred_key_points_during_generate = []
                if self.config.kp_decoder_type == "candi_cls":
                    assert key_points_num == 1, f"candi_cls only support key_points_num == 1 bug get {key_points_num}"
                    assert self.config.task == "nuplan", f"only suport nuplan task for candi_cls"
                    input_embeds_current = input_embeds[:, :kp_start_index, :]
                    attention_mask = torch.ones(input_embeds_current.shape[:2], dtype=torch.long, device=input_embeds.device)
                    position_ids = self._prepare_position_ids_for_generation(attention_mask.clone())
                    transformer_outputs_hidden_state = self.embedding_to_hidden(
                        input_embeds_current,
                        attention_mask,
                        position_ids,
                    )
                    pred_key_points_ids = []
                    pred_key_points_scores = []
                    pred_key_points_topk = []
                    pred_key_points_topk_scores = []

                    future_key_point_hidden_state = transformer_outputs_hidden_state[:, kp_start_index - candi_kp_num:kp_start_index, :]

                    key_point_ids, key_points_scores, key_point_ids_topk, key_point_scores_topk = self.key_points_decoder[0].generate_keypoints(future_key_point_hidden_state, self.config.kp_decoder_num)
                    key_points_logit = self.kp_tokenizer[0].decode(key_point_ids, dtype=key_points_scores.dtype, device=key_points_scores.device).unsqueeze(1)
                    pred_key_points_ids.append(key_point_ids) # [ (bs,) ]
                    pred_key_points_scores.append(key_points_scores) # [ (bs, 1, n_cluster)]

                    key_points_score_top_ks = []
                    key_points_logit_top_ks = []
                    for top_k in range(key_point_ids_topk.shape[-1]):
                        # [topk (bs, 2) ]
                        key_points_logit_top_ks.append(self.kp_tokenizer[0].decode(key_point_ids_topk[:, top_k], dtype=key_points_scores.dtype, device=key_points_scores.device) )
                        key_points_score_top_ks.append(key_point_scores_topk[:, top_k]) # [topk (bs,)]
                    pred_key_points_topk.append(key_points_logit_top_ks) # [K [topk (bs, 2) ]]
                    pred_key_points_topk_scores.append(key_points_score_top_ks) # [K [topk (bs,)]

                    pred_key_point = key_points_logit

                    key_point_embed = self.encoder.kps_m_embed(pred_key_point).reshape(batch_size, 1, -1)  # b, 1, n_embed
                    input_embeds[:, kp_start_index, :] = key_point_embed[:, 0, :]
                    pred_key_points_during_generate.append(pred_key_point[:, 0, :2].unsqueeze(1))
                else:
                    for i in range(key_points_num):
                        input_embeds_current = input_embeds[:, :kp_start_index + i, :]
                        attention_mask = torch.ones(input_embeds_current.shape[:2], dtype=torch.long, device=input_embeds.device)
                        position_ids = self._prepare_position_ids_for_generation(attention_mask.clone())
                        transformer_outputs_hidden_state = self.embedding_to_hidden(
                            input_embeds_current,
                            attention_mask,
                            position_ids,
                        )['last_hidden_state']
                        future_key_point_hidden_state = transformer_outputs_hidden_state[:,
                                                        kp_start_index + i - 1,
                                                        :].reshape(batch_size, 1, -1)
                        if self.kp_tokenizer is None:
                            key_points_logit, _ = self.key_points_decoder.generate_keypoints(future_key_point_hidden_state)
                        elif self.config.regression_long_class_short and i in [0, 1, 2]:
                            key_points_logit, _ = self.key_points_decoder[i].generate_keypoints(future_key_point_hidden_state)
                        else:
                            key_point_ids, key_points_scores = self.key_points_decoder[i].generate_keypoints(future_key_point_hidden_state)
                            key_points_logit = self.kp_tokenizer[i].decode(key_point_ids, dtype=key_points_scores.dtype, device=key_points_scores.device).unsqueeze(1)

                        if self.k > 1:
                            k_key_points_logit = key_points_logit['logits']  # (bs, 1, k, 2/4)
                            k_key_points_scores = key_points_logit['scores']  # (bs, 1, k)
                            _, seq_len, _, last_dim = k_key_points_logit.shape  # seq_len = 1 per key point
                            selected_key_points = []
                            for j in range(batch_size):
                                selected_key_points_current_batch = []
                                for k in range(seq_len):
                                    top_score, top_indx = torch.topk(k_key_points_scores[j, k, :], dim=-1, k=1)
                                    selected_key_points_current_batch.append(k_key_points_logit[j, k, top_indx[0], :])
                                selected_key_points_current_batch = torch.stack(selected_key_points_current_batch, dim=0)
                                selected_key_points.append(selected_key_points_current_batch)  # a list of (1, 2/4)
                            key_points_logit = torch.stack(selected_key_points, dim=0)  # (bs, 1, 2/4)

                        if gt_1s_kp is not None and i == 3:
                            # assert False, 'deprecated, debug only'
                            print('testing with gt1skp: ', key_points_logit[0, 0, :2] - gt_1s_kp[0, 0, :2])
                            key_points_logit = gt_1s_kp

                        # pred_key_point = torch.zeros((batch_size, 1, 2), device=device)
                        pred_key_point = key_points_logit

                        off_road_checking = True
                        if off_road_checking and route_ids is not None and road_dic is not None and ego_pose is not None and map_name is not None:
                            from transformer4planning.utils import nuplan_utils
                            for sample_index in range(batch_size):
                                # if i in [0, 1] and 'backward' in self.use_key_points:
                                # Check key points with map_api
                                # WARNING: WIP, do not use
                                y_inverse = -1 if map_name[sample_index] == 'sg-one-north' else 1
                                pred_key_point_copy = copy.deepcopy(pred_key_point[sample_index, 0, :2])
                                pred_key_point_copy[1] *= y_inverse
                                pred_key_point_global = nuplan_utils.change_coordination(pred_key_point_copy[:2].cpu().numpy(),
                                                                                        ego_pose[sample_index],
                                                                                        ego_to_global=True)
                                if isinstance(route_ids[sample_index], torch.Tensor):
                                    route_ids_this_sample = route_ids[sample_index].cpu().numpy().tolist()
                                else:
                                    route_ids_this_sample = route_ids[sample_index]
                                route_ids_this_sample = [int(route_id) for route_id in route_ids_this_sample]
                                closest_lane_point_on_route, dist, _, _, on_road = nuplan_utils.get_closest_lane_point_on_route(pred_key_point_global,
                                                                                                                                route_ids_this_sample,
                                                                                                                                road_dic[sample_index])
                                if not on_road:
                                    # changing to lane center from 4? to 53, average 51
                                    revised_pred_point = closest_lane_point_on_route
                                    pred_key_point_ego = nuplan_utils.change_coordination(revised_pred_point,
                                                                                        ego_pose[sample_index],
                                                                                        ego_to_global=False)
                                    pred_key_point_ego[1] *= y_inverse
                                    pred_key_point[sample_index, 0, :2] = torch.tensor(pred_key_point_ego, device=pred_key_point.device)
                                    print(f'Off Road Detected! Replace {i}th key point')

                        object_collision_checking = False
                        agents_rect_local = kwargs.get('agents_rect_local', None)
                        if object_collision_checking and agents_rect_local is not None:
                            assert self.config.selected_exponential_past, 'only support selected_exponential_past for now'

                            # flip the second and third dimension
                            agents_rect_local = agents_rect_local.permute(0, 2, 1, 3, 4)  # (b, 300, 4, 4, 2)
                            ego_shape = [2.297, 5.176]
                            for sample_index in range(batch_size):
                                if i in [0, 1]:
                                    break
                                agents_rect_local_this_sample = agents_rect_local[sample_index]
                                ego_center = pred_key_point[sample_index, 0, :2].float().cpu().numpy()
                                from shapely import geometry
                                ego_rect = geometry.box(ego_center[0] - ego_shape[0] / 2, ego_center[1] - ego_shape[0] / 2,
                                                        ego_center[0] + ego_shape[0] / 2, ego_center[1] + ego_shape[0] / 2)
                                for each_agent_index in range(agents_rect_local_this_sample.shape[0]):
                                    agent_rects = agents_rect_local_this_sample[
                                        each_agent_index].float().cpu().numpy()  # (4(past_steps), 4(box), 2(x,y))
                                    if agent_rects.sum() == 0:
                                        # padding numbers
                                        continue
                                    # only process static objects
                                    future_position_at_t = agent_rects[0, :, :]
                                    # create polygon for the future position
                                    try:
                                        future_line = geometry.LineString(future_position_at_t)
                                        future_poly = geometry.Polygon(future_line)
                                    except:
                                        print('future_position_at_t failed to create polygon: ', future_position_at_t)
                                        continue
                                    if future_poly.intersects(ego_rect):
                                        # replace key point with a slower speed
                                        speed_penalty_rate = 0.8
                                        pred_key_point[sample_index, 0, :2] *= speed_penalty_rate
                                        print(f'Object Collision Detected! Replace {i}th key point')
                                        print(f'ego: {ego_center}, agent: {future_position_at_t}')
                                        break

                        if self.config.task == "nuplan":
                            if not self.config.separate_kp_encoder:
                                assert False, 'deprecated, use separate_kp_encoder instead'
                            # if self.config.use_speed:
                            #     # padding speed, padding the last dimension from 4 to 7
                            #     pred_key_point = torch.cat([pred_key_point, torch.zeros_like(pred_key_point)[:, :, :3]], dim=-1)
                            # key_point_embed = self.encoder.action_m_embed(pred_key_point).reshape(batch_size, 1, -1)  # b, 1, n_embed
                            else:
                                key_point_embed = self.encoder.kps_m_embed(pred_key_point).reshape(batch_size, 1, -1)  # b, 1, n_embed
                        else:
                            assert False, 'Key Point for waymo not implemented yet'
                        # replace embed at the next position
                        input_embeds[:, kp_start_index + i, :] = key_point_embed[:, 0, :]
                        pred_key_points_during_generate.append(pred_key_point[:, 0, :2].unsqueeze(1))

                key_points_logits = torch.cat(pred_key_points_during_generate, dim=1).reshape(batch_size, key_points_num, -1)
                key_points_logits_k.append(key_points_logits)

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!plot cluster key-point
            # #plot key-point
            # pred_key_points_ids = torch.stack(pred_key_points_ids, dim=1) # [(bs,),... ] -> (bs, K)
            # # pred_key_points_scores = torch.cat(pred_key_points_scores, dim=1) # [ (bs, 1, n_cluster), ...] -> (bs, K, n_cluster)

            # bs, K = pred_key_points_ids.shape
            # gt_traj = info_dict['trajectory_label'][:, :, :2].cpu().numpy() # bs,80,2
            # gt_future_key_points = info_dict['future_key_points'].cpu().numpy() # bs,K,4
            # # gt_future_key_points_aug = info_dict['future_key_points_aug'].cpu().numpy() # bs,K,2
            # gt_future_key_points_ids = info_dict['future_key_points_ids'].cpu().numpy() # bs,K
            # gt_future_key_points_qat = info_dict['future_key_points_after'].cpu().numpy() # bs,K,2
            # pred_future_key_points_ids = pred_key_points_ids.cpu().numpy() # (bs,K)
            # pred_future_key_points_qat = key_points_logits.cpu().numpy() # bs,K,2
            # pred_future_key_points_scores = pred_key_points_scores #.cpu().numpy() # (bs,K,n_cluster)
            # pred_future_key_points_topk = pred_key_points_topk #[K [topk (bs,2), ] ]
            # pred_future_key_points_topk_scores = pred_key_points_topk_scores # [K [topk (bs,)]

            # import matplotlib.pyplot as plt
            # import os
            # for bs_i in range(bs):
            #     plt.figure(figsize=(10, 6))
            #     score_str = ""
            #     alphas = [1.0, 0.8, 0.6, 0.4, 0.2]
            #     sizes = [4, 6, 8, 10, 12]
            #     for k in range(K):
            #         # gt_centers = self.kp_tokenizer[k].centers.cpu().numpy() # n_cluster,2
            #         gt_kp = gt_future_key_points[bs_i, k, :2]
            #         # gt_kp_aug = gt_future_key_points_aug[bs_i, k, :2]
            #         gt_kp_id = gt_future_key_points_ids[bs_i, k]
            #         gt_kp_qat = gt_future_key_points_qat[bs_i, k, :2]
            #         pred_kp_id = pred_future_key_points_ids[bs_i, k]
            #         pred_kp_score = pred_future_key_points_scores[k].cpu().numpy()[bs_i, pred_kp_id]
            #         pred_kp_qat = pred_future_key_points_qat[bs_i, k, :2]

            #         # plt.scatter(
            #         #     x=gt_centers[:,0],
            #         #     y=gt_centers[:,1],
            #         #     # s=df['label_num'],
            #         #     alpha=0.2,
            #         #     c='k',
            #         #     cmap='viridis'
            #         # )

            #         # print("gt_kp", gt_kp)
            #         # print("gt_kp_aug", gt_kp_aug)
            #         # print("gt_kp_qat", gt_kp_qat)
            #         # print("pred_kp_qat", pred_kp_qat)
            #         plt.plot(gt_traj[bs_i, :, 0], gt_traj[bs_i, :, 1], '-', color='g')
            #         plt.plot(gt_kp[0], gt_kp[1], 'o', markerfacecolor='none', markeredgecolor='g', markersize=sizes[k], alpha=alphas[k])
            #         # plt.plot(gt_kp_aug[0], gt_kp_aug[1],'+', markerfacecolor='none', markeredgecolor='b', markersize=sizes[k], alpha=alphas[k])
            #         plt.plot(gt_kp_qat[0], gt_kp_qat[1],'x', markerfacecolor='none', markeredgecolor='y', markersize=sizes[k], alpha=alphas[k])
            #         plt.plot(pred_kp_qat[0], pred_kp_qat[1], 'o', markerfacecolor='none', markeredgecolor='r', markersize=sizes[k], alpha=alphas[k])
            #         score_str += f' K:{k} kp_id:{gt_kp_id} vs {pred_kp_id} score:{pred_kp_score:.2f}'

            #         # for topk
            #         for topk in range(6):
            #             topk_pred_kp = pred_future_key_points_topk[k][topk].cpu().numpy()[bs_i]
            #             topk_pred_kp_score = pred_future_key_points_topk_scores[k][topk].cpu().numpy()[bs_i]
            #             if topk == 0:
            #                 assert topk_pred_kp_score == pred_kp_score, f"k={topk} topk_pred_kp_score:{topk_pred_kp_score} != pred_kp_score:{pred_kp_score}"
            #                 continue
            #             plt.plot(topk_pred_kp[0], topk_pred_kp[1], '*', markerfacecolor='none', markeredgecolor='r', markersize=sizes[k], alpha=alphas[k])
            #             score_str += f" top_{topk}:{topk_pred_kp_score:.2f}"

            #         if k%2 == 0:
            #             score_str += "\n"



            #     # plt.xlabel('label_center_x')
            #     # plt.ylabel('label_center_y')
            #     plt.title(score_str, fontsize=6)
            #     plt.axis("equal")
            #     #save_path = "debug/CKS-Small-Cluster-1024/"
            #     #save_path = "debug/CKS-Small-Uniform/"
            #     save_path = "debug/"
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     plt.savefig(os.path.join(save_path, f"{kwargs['file_name'][bs_i]}_frameid_{kwargs['frame_id'][bs_i]}.png"))
            #     plt.close()

            # generate remaining trajectory
            transformer_outputs_hidden_state = self.embedding_to_hidden(input_embeds)['last_hidden_state']
            if self.config.autoregressive:
                points_to_predict = info_dict["pred_length"]
                raster_seq_length = info_dict["sequence_length"]
                traj_logits = torch.zeros((batch_size, points_to_predict, 4), device=device)
                predicting_index = context_length - 1  # output space index
                for i in range(points_to_predict):
                    # 1. generate trajectory
                    current_context_length = context_length + i * (1 + raster_seq_length)  # input space index
                    context_embeds = input_embeds[:, :current_context_length, :]
                    attention_mask = torch.ones(context_embeds.shape[:2], dtype=torch.long, device=device)
                    position_ids = self._prepare_position_ids_for_generation(attention_mask.clone())
                    transformer_outputs_hidden_state = self.embedding_to_hidden(
                        context_embeds,
                        attention_mask,
                        position_ids,
                    )['last_hidden_state']
                    traj_logits[:, i, :] = self.traj_decoder.model(transformer_outputs_hidden_state)[:, predicting_index, :]
                    predicting_index += (1 + raster_seq_length)
                    # 2. update input_embeds with generated trajectory
                    new_point_embed = self.encoder.action_m_embed(traj_logits[:, i, :].unsqueeze(1))
                    input_embeds[:, current_context_length, :] = new_point_embed[:, 0, :]
                    # 3. generate observation
                    if i == points_to_predict - 1:
                        # skip generating observations for the last point
                        continue
                    # TODO: change to generate observation
                    next_state_embeds = torch.zeros((batch_size, raster_seq_length, 1, self.config.n_embd), device=device)
                    for j in range(raster_seq_length):
                        input_embeds[:, current_context_length + 1 + j, :] = next_state_embeds[:, j, 0, :]
                traj_logits_k.append(traj_logits)
                # # expand traj_logits from (b, 8, 4/7) to (b, 80, 4/7) with linear interpolation
                # expanded_traj_logits = torch.zeros((batch_size, points_to_predict * 10, 4), device=device)
                # # add padding 0 to the first point to the traj_logits for linear interpolation
                # traj_logits = torch.cat([torch.zeros((batch_size, 1, 4), device=device), traj_logits], dim=1)
                # for i in range(points_to_predict):
                #     for j in range(10):
                #         expanded_traj_logits[:, i * 10 + j, :] = traj_logits[:, i, :] + (traj_logits[:, i + 1, :] - traj_logits[:, i, :]) * j / 10
                # traj_logits_k.append(expanded_traj_logits)
            # expected shape for pred trajectory is (b, pred_length, 4/7)
            elif self.config.kp_tokenizer == 'cluster_traj':
                traj_logits, scores = self.key_traj_decoder.generate_trajs(transformer_outputs_hidden_state, info_dict)
                if self.config.reverse_traj_index_order:
                    traj_logits = traj_logits.flip(-2)
                traj_logits_k.append(traj_logits)
            elif self.traj_decoder is not None:
                traj_logits = self.traj_decoder.generate_trajs(transformer_outputs_hidden_state, info_dict)
                if self.config.reverse_traj_index_order:
                    traj_logits = traj_logits.flip(-2)
                traj_logits_k.append(traj_logits)
                if self.config.kp_decoder_type == "candi_cls":
                    for i in range(len(pred_key_points_topk[0])):
                        if i==0:
                            continue
                        key_point_embed = self.encoder.kps_m_embed(pred_key_points_topk[0][i]).reshape(batch_size, 1, -1)  # b, 1, n_embed

                        input_embeds[:, kp_start_index, :] = key_point_embed[:, 0, :]
                        pred_key_points_during_generate.append(pred_key_point[:, 0, :2].unsqueeze(1))
                        transformer_outputs_hidden_state = self.embedding_to_hidden(input_embeds)
                        traj_logits = self.traj_decoder.generate_trajs(transformer_outputs_hidden_state, info_dict)
                        traj_logits_k.append(traj_logits)
                    key_points_logits = torch.cat(pred_key_points_during_generate, dim=1).reshape(batch_size, self.config.kp_decoder_num, -1)
                    key_points_logits_k = []
                    key_points_logits_k.append(key_points_logits)
            else:
                raise NotImplementedError

        key_points_pred_logits = None
        if select_k == 1:
            if self.config.kp_tokenizer == "cluster_traj":
                traj_pred_logits = traj_logits_k[0]
            elif self.config.kp_decoder_type == "candi_cls":
                traj_pred_logits = torch.stack(traj_logits_k, dim=1)
            else:
                traj_pred_logits = traj_logits_k[0]
            if len(key_points_logits_k) > 0:
                # WARNING, k select if not implemented for key points
                assert len(key_points_logits_k) == select_k
                key_points_pred_logits = key_points_logits_k[0]
        else:
            traj_pred_logits = torch.stack(traj_logits_k, dim=1)
            if len(key_points_logits_k) > 0:
                assert len(key_points_logits_k) == self.k
                key_points_pred_logits = torch.stack(key_points_logits_k, dim=1)

        pred_dict = {
            "traj_logits": traj_pred_logits
        }

        # if kwargs.get('old_file_name', None) is not None:
        #     logger.warning('passing gt at gen next step for debugging')
        #     pred_dict['traj_logits'] = kwargs['trajectory_label']

        if key_points_pred_logits is not None:
            pred_dict.update({"key_points_logits": key_points_pred_logits})

        if self.use_proposal:
            pred_dict.update({"proposal": proposal_result})  # topk results
            pred_dict.update({"proposal_scores": proposal_scores})  # topk scores
            if self.config.task == 'nuplan' and 'intentions' in info_dict:
                pred_dict.update({'intentions': info_dict['intentions']})

        if self.config.task == "waymo":
            center_objects_world = kwargs['center_objects_world'].type_as(traj_pred_logits)
            num_center_objects, num_modes, num_timestamps, num_feat = traj_pred_logits.shape

            from transformer4planning.utils.waymo_utils import rotate_points_along_z, str_to_tensor

            pred_trajs_world = rotate_points_along_z(
                points=traj_pred_logits.view(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].view(num_center_objects)
            ).view(num_center_objects, num_modes, num_timestamps, num_feat)
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]

            pred_dict = {
                'scenario_id': str_to_tensor(kwargs['scenario_id']).to(device),
                'pred_trajs': pred_trajs_world[:, :, :, 0:2],
                'pred_scores': topk_score,
                'object_id': torch.tensor(kwargs['center_objects_id']).to(device),
                'object_type': torch.tensor(kwargs['center_objects_type']).to(device),
                'gt_trajs': kwargs['center_gt_trajs_src'],
                'track_index_to_predict': kwargs['track_index_to_predict'],
            }

        return pred_dict


def build_models(model_args):
    # TODO: refactor model building function into each model class
    if 'gpt' in model_args.model_name:
        from transformer4planning.models.backbone.gpt2 import STR_GPT2, STRGPT2Config
        config_p = STRGPT2Config()
        config_p.update_by_model_args(model_args)
        # from transformer4planning.models.backbone.gpt2 import TrajectoryGPT
        ModelCls = STR_GPT2
        tag = 'GPTTrajectory'
        if 'gpt-mini' in model_args.model_name:
            """
            Number of parameters: 300k
            """
            config_p.n_layer = 1
            config_p.n_embd = config_p.d_model = 64
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 1
        elif 'gpt-small' in model_args.model_name:
            """
            Number of parameters: 16M
            """
            config_p.n_layer = 4
            config_p.n_embd = config_p.d_model = 256
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 8
        elif 'gpt-medium' in model_args.model_name:
            """
            Number of parameters: 124M
            """
            config_p.n_layer = 12
            config_p.n_embd = config_p.d_model = 768
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 12
        elif 'gpt-large' in model_args.model_name:
            """
            Number of parameters: 1.5B
            """
            config_p.n_layer = 48
            config_p.n_embd = config_p.d_model = 1600
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 25
        else:
            logger.warning('Using default GPT2 config')
            config_p.n_layer = model_args.n_layers
            config_p.n_embd = model_args.d_embed
            config_p.n_inner = model_args.d_inner
            config_p.n_head = model_args.n_heads
        config_p.activation_function = model_args.activation_function

        if model_args.task == "train_diffusion_decoder":
            from transformer4planning.models.decoder.diffusion_decoder import (KeypointDiffusionModel, T4PTrainDiffWrapper)
            out_features = 4 if model_args.predict_yaw else 2
            diffusion_model = KeypointDiffusionModel(config_p.n_inner,
                                                     config_p.n_embd,
                                                     out_features=out_features,
                                                     key_point_num=1,
                                                     input_feature_seq_lenth=model_args.diffusion_condition_sequence_lenth,
                                                     use_key_points=model_args.use_key_points,
                                                     feat_dim=model_args.key_points_diffusion_decoder_feat_dim,)
            model = T4PTrainDiffWrapper(diffusion_model, num_key_points=model_args.key_points_num, model_args=config_p)
            if model_args.key_points_diffusion_decoder_load_from is not None:
                state_dict = torch.load(model_args.key_points_diffusion_decoder_load_from)
                model.load_state_dict(state_dict)
                logger.info("Pretrained keypoint decoder has been loaded!")
            logger.info("Only diffusion decoder will be trained!")
            return model
        # whole model training
    elif 'mamba' in model_args.model_name:
        from transformer4planning.models.backbone.mamba import STRMamba, STRMambaConfig
        config_p = STRMambaConfig()
        config_p.update_by_model_args(model_args)
        ModelCls = STRMamba
        tag = 'MambaTrajectory'
        if 'mamba-mini' in model_args.model_name:
            """
            Number of parameters: ?
            """
            config_p.n_layer = 1
            config_p.n_embd = config_p.d_model = 64
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 1
        elif 'mamba-small' in model_args.model_name:
            """
            Number of parameters: 6M (ViT512)
            """
            config_p.n_layer = 4
            config_p.n_embd = config_p.d_model = 256
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 8
        elif 'mamba-medium' in model_args.model_name:
            """
            Number of parameters: 27M (ViT)
            """
            config_p.n_layer = 8
            config_p.n_embd = config_p.d_model = 512
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 16
        elif 'mamba-large' in model_args.model_name:
            """
            WARNING: Gradient WILL CRUSH DURING TRAINING
            Number of parameters: 139M (ViT)
            """
            config_p.n_layer = 16
            config_p.n_embd = config_p.d_model = 1000
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 25
        elif 'mamba-xl' in model_args.model_name:
            """
            Number of parameters: 764M (ViT)
            """
            config_p.n_layer = 24
            config_p.n_embd = config_p.d_model = 2048
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 32
    elif 'mixtral' in model_args.model_name:
        from transformer4planning.models.backbone.mixtral import STR_Mixtral, STRMixtralConfig
        config_p = STRMixtralConfig()
        config_p.update_by_model_args(model_args)
        ModelCls = STR_Mixtral
        tag = 'MixTralTrajectory'
        if 'mixtral-small-wide' in model_args.model_name:
            config_p.n_layer = 4
            config_p.n_embd = config_p.d_model = 512
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 8
            config_p.num_hidden_layers = 4
            config_p.hidden_size = 512
            config_p.intermediate_size = config_p.n_embd * 4
            config_p.num_attention_heads = 8
        elif 'mixtral-small' in model_args.model_name:
            config_p.n_layer = 4
            config_p.n_embd = config_p.d_model = 256
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 8
            config_p.num_hidden_layers = 4
            config_p.hidden_size = 256
            config_p.intermediate_size = config_p.n_embd * 4
            config_p.num_attention_heads = 8
        elif 'mixtral-narrow-medium' in model_args.model_name:
            """
            Number of parameters: 1.2B with 24 experts with top 2 (ViT)
            """
            config_p.n_layer = 16
            config_p.n_embd = config_p.d_model = 512
            config_p.n_inner = 2048
            config_p.n_head = 16
            config_p.num_hidden_layers = 16
            config_p.hidden_size = 512
            config_p.intermediate_size = 2048
            config_p.num_attention_heads = 16
        elif 'mixtral-medium' in model_args.model_name:
            """
            Number of parameters: 1.5B (ViT)
            """
            config_p.n_layer = 16
            config_p.n_embd = config_p.d_model = 1024
            config_p.n_inner = 4096
            config_p.n_head = 16
            config_p.num_hidden_layers = 16
            config_p.hidden_size = 1024
            config_p.intermediate_size = 4096
            config_p.num_attention_heads = 16
        elif 'mixtral-800m-deep' in model_args.model_name:
            # 800M trainable with batch size of 32
            config_p.n_layer = 32
            config_p.n_embd = config_p.d_model = 512
            config_p.n_inner = 2048
            config_p.n_head = 32
            config_p.hidden_size = config_p.n_embd
            config_p.intermediate_size = config_p.n_inner
            config_p.num_attention_heads = config_p.n_head
        elif 'mixtral-3b-deep' in model_args.model_name:
            """
            Number of parameters: 350M x 8 -> 3.3B (ViT)
            """
            # 3.3B trainable with batch size of 8
            config_p.n_layer = 32
            config_p.n_embd = config_p.d_model = 1024
            config_p.n_inner = 4096
            config_p.n_head = 32
            config_p.hidden_size = config_p.n_embd
            config_p.intermediate_size = config_p.n_inner
            config_p.num_attention_heads = config_p.n_head
        elif 'mixtral-3b-wide' in model_args.model_name:
            """
            Number of parameters: 350M x 8 -> 3.8B (ViT)
            Previous Large
            """
            # 3.8B trainable with batch size of 8
            config_p.n_layer = 16
            config_p.n_embd = config_p.d_model = 1536
            config_p.n_inner = 6144
            config_p.n_head = 16
            config_p.num_hidden_layers = config_p.n_layer
            config_p.hidden_size = config_p.n_embd
            config_p.intermediate_size = config_p.n_inner
            config_p.num_attention_heads = config_p.n_head
        elif 'mixtral-6b-deep' in model_args.model_name:
            """
            Number of parameters: 6B (ViT)
            """
            # 6B trainable
            # config_p.n_layer = 25
            # config_p.n_embd = config_p.d_model = 1536
            # config_p.n_inner = 6144
            # config_p.n_head = 16

            # 6.5B trainable
            # config_p.n_layer = 20
            # config_p.n_embd = config_p.d_model = 1792
            # config_p.n_inner = 7168
            # config_p.n_head = 16

            # 6.7B trainable with batch size of 1
            config_p.n_layer = 64
            config_p.n_embd = config_p.d_model = 1024
            config_p.n_inner = 4096
            config_p.n_head = 64
            config_p.num_hidden_layers = config_p.n_layer
            config_p.hidden_size = config_p.n_embd
            config_p.intermediate_size = config_p.n_inner
            config_p.num_attention_heads = config_p.n_head
        else:
            assert False, f'Unsupported model name: {model_args.model_name}!'
    elif 'stablelm' in model_args.model_name:
        from transformer4planning.models.backbone.stablelm import STR_StableLM, STRStableLMConfig
        config_p = STRStableLMConfig()
        config_p.update_by_model_args(model_args)
        ModelCls = STR_StableLM
        tag = 'StableLMTrajectory'
        if 'stablelm-small' in model_args.model_name:
            """
            Number of parameters: 22M (ViT)
            """
            config_p.n_layer = 4
            config_p.n_embd = config_p.d_model = 256
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 8
            config_p.num_hidden_layers = 4
            config_p.hidden_size = 256
            config_p.intermediate_size = config_p.n_embd * 4
            config_p.num_key_value_heads = config_p.num_attention_heads = 8
            attn_implementation = "flash_attention_2"
        elif 'stablelm-medium' in model_args.model_name:
            """
            Number of parameters: 300M (ViT)
            """
            config_p.n_layer = 16
            config_p.n_embd = config_p.d_model = 1024
            config_p.n_inner = 4096
            config_p.n_head = 16
            config_p.num_hidden_layers = 16
            config_p.hidden_size = 1024
            config_p.intermediate_size = 4096
            config_p.num_key_value_heads = config_p.num_attention_heads = 16
        elif 'stablelm-large' in model_args.model_name:
            """
            WARNING: Gradient WILL CRUSH DURING TRAINING
            Number of parameters: 350M x 8 -> 2,2B (ViT)
            """
            config_p.n_layer = 24
            config_p.n_embd = config_p.d_model = 1024
            config_p.n_inner = 3584
            config_p.n_head = 16
            config_p.num_hidden_layers = 24
            config_p.hidden_size = 1024
            config_p.intermediate_size = 3584
            config_p.num_key_value_heads = config_p.num_attention_heads = 16
        elif 'stablelm-xl' in model_args.model_name:
            """
            WARNING: Gradient WILL CRUSH DURING TRAINING
            Number of parameters: 350M x 8 -> 2,2B (ViT)
            """
            config_p.n_layer = 24
            config_p.n_embd = config_p.d_model = 2048
            config_p.n_inner = 7168
            config_p.n_head = 16
            config_p.num_hidden_layers = 24
            config_p.hidden_size = 2048
            config_p.intermediate_size = 7168
            config_p.num_key_value_heads = config_p.num_attention_heads = 16
    else:
        raise ValueError("Model name must choose from ['scratch', 'pretrain'] + ['nonauto-gpt', 'transxl', 'gpt', 'xlnet']!")

    if 'scratch' in model_args.model_name:
        model = ModelCls(config_p)
        logger.info('Scratch ' + tag + ' Initialized!')
    elif 'pretrain' in model_args.model_name:
        # from transformers import AqlmConfig
        # quantization_config = AqlmConfig(weights="int8", pre_quantized=False)
        model = ModelCls.from_pretrained(model_args.model_pretrain_name_or_path, config=config_p)
        logger.info('Pretrained ' + tag + ' from {}'.format(model_args.model_pretrain_name_or_path))
        if model_args.key_points_diffusion_decoder_load_from is not None:
                print(f"Now loading pretrained key_points_diffusion_decoder from {model_args.key_points_diffusion_decoder_load_from}.")
                state_dict = torch.load(model_args.key_points_diffusion_decoder_load_from)
                model.key_points_decoder.model.load_state_dict(state_dict)
    elif 'transfer' in model_args.model_name:
        model = ModelCls(config_p)
        logger.info('Transfer' + tag + ' from {}'.format(model_args.model_pretrain_name_or_path))
    return model


def build_model_from_path(model_path):
    from transformer4planning.utils.args import ModelArguments
    from transformers import (HfArgumentParser)
    import os
    parser = HfArgumentParser((ModelArguments))
    # load model args from config.json
    config_path = os.path.join(model_path, 'config.json')
    if not os.path.exists(config_path):
        print('WARNING config.json not found in checkpoint path, using default model args ', config_path)
        model_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    else:
        model_args, = parser.parse_json_file(config_path, allow_extra_keys=True)
        model_args.model_pretrain_name_or_path = model_path
        model_args.model_name = model_args.model_name.replace('scratch', 'pretrained')
    print('model loaded with args: ', model_args, model_args.model_name, model_path)
    model = build_models(model_args=model_args)
    return model


def interpolate_yaw(pred_traj, mode, yaw_change_upper_threshold=0.1):
    # deprecated
    if mode == "normal":
        return pred_traj
    elif mode == "interplate" or mode == "hybrid":
        # Warning: this function is tested not better than normal mode
        assert False, "Warning: this function is tested not better than normal mode, to be updated in the future"
        # generating yaw angle from relative_traj
        dx = pred_traj[:, 4::5, 0] - pred_traj[:, :-4:5, 0]
        dy = pred_traj[:, 4::5, 1] - pred_traj[:, :-4:5, 1]
        distances = torch.sqrt(dx ** 2 + dy ** 2)
        relative_yaw_angles = torch.where(distances > 0.1, torch.arctan2(dy, dx), 0)
        # accumulate yaw angle
        # relative_yaw_angles = yaw_angles.cumsum()
        relative_yaw_angles_full = relative_yaw_angles.repeat_interleave(5, dim=1)
        if mode == "interplate":
            pred_traj[:, :, -1] = relative_yaw_angles_full
        else:
            pred_traj[:, :, -1] = torch.where(torch.abs(pred_traj[:, :, -1]) > yaw_change_upper_threshold, relative_yaw_angles_full, pred_traj[:, :, -1])
    return pred_traj
