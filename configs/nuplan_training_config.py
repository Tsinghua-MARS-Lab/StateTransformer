from pathlib import Path
from rl_frame.model.components.mingpt import GPTConfig

class Config(object):
    def __init__(self):
        self.window_w = 1200
        self.window_h = 1200
        self.scale = 5
        self.ego_centric = True
        self.show_trajectory = True

        self.dataset = 'NuPlan'
        self.running_mode = 0
        self.render_to_cv = False
        self.render_to_tk = True

        self.save_log_every_scenario = True

        self.dynamic_env_planner = 'env'  # pass in False for an open-loop simulation
        # other default ego planners: None (None for playback), 'dummy', 'trajpred', 'e2e'
        self.ego_planner = 'opt'
        # self.ego_planner = None
        # self.dynamic_env_planner = None
        if self.dynamic_env_planner == 'env' and self.ego_planner in {'base', 'opt'}:
            # parameters for default planners
            self.predict_env_for_ego_collisions = None  # or 'M2I'
            self.predict_relations_for_env = True
            self.predict_relations_for_ego = True
            self.predict_with_rules = True
        else:
            # parameters for default planners
            self.predict_env_for_ego_collisions = 'M2I'
            self.predict_relations_for_env = False
            self.predict_relations_for_ego = False
            self.predict_with_rules = True

        self.planning_task = '8s'
        if self.dataset == 'Waymo':
            # Waymo
            self.frame_rate = 10
            self.planning_from = 11
            self.planning_to = 90
            self.planning_horizon = 80
            self.total_frame = 91
            self.planning_interval = 10  # 5
            # self.tf_example_dir = str(Path.home()) + '/waymo_data/tf_validation_interact'
            # self.tf_example_dir = str(Path.home()) + '/waymo_data/training_interactive'
            self.tf_example_dir = str(Path.home()) + '/waymo_data/training'
            self.map_name = 'Inter.Val'  # used for log simulation info
        if self.dataset == 'NuPlan':
            # NuPlan
            self.frame_rate = 20
            self.planning_from = 40
            self.planning_to = 181
            self.planning_horizon = 160
            self.total_frame = 181  # 91


            self.planning_to = 240
            self.planning_horizon = 160
            self.total_frame = 241  # 91

            self.planning_interval = 10  # 5
            self.data_path = {
                'NUPLAN_DATA_ROOT': str(Path.home()) + "/nuplan/dataset",
                'NUPLAN_MAPS_ROOT': str(Path.home()) + "/nuplan/dataset/maps",
                'NUPLAN_DB_FILES': str(Path.home()) + "/nuplan/dataset/nuplan-v1.0/public_set_boston_train/",
            }
            self.map_name = 'Boston'  # used for log simulation info

        self.predictor = 'M2I'

        # Irrelevant configs
        self.loaded_prediction_path = '0107.multiInf_last8_vectorInf_tfR_reactorPred_all.pickle'
        self.draw_prediction = False  # for debugging loaded prediction trajectories
        self.draw_collision_pt = False  # for debugging loaded collision pts
        self.load_prediction_with_offset = True

        self.model_path = {
            'guilded_m_pred': './prediction/M2I/guilded_m_pred/pretrained/LowSpeedGoalDn.model.30.bin',
            'marginal_pred': './prediction/M2I/marginal_prediction/pretrained/8S.raster.maskNonInfFilterSteadyP5.Loopx10.0410.model.9.bin',
            'relation_pred': './prediction/M2I/relation_predictor/pretrained/infPred.NoTail.timeOffset.loopx2.IA.v2x.NonStop20.noIntention.0424.model.60.bin',
            'variety_loss_prediction': './prediction/M2I/variety_loss_prediction/pretrained/model.26.bin',
            'marginal_pred_tnt': './prediction/M2I/marginal_prediction_tnt/pretrained/tnt.model.21.bin'
        }

        self.test_task = 2  # 0=collision solving, 1=trajectory quality, 2=baseline ego planner
        self.testing_method = 2  # 0=densetnt with dropout, 1=0+post-processing, 2=1+relation, -1=variety loss

        self.filter_static = False
        self.filter_non_vehicle = False
        self.all_relevant = False
        self.follow_loaded_relation = False
        self.follow_prediction = False
        self.follow_gt_first = False  # do not use, unstable switching back and forward

        # test for upper bound
        self.follow_gt_relation = False

        self.playback_dir = None

        self.scenario_augment = {
            'mask_non_influencer': [1, 100],  # [min, max]
            'mask_influencer': [1, 10],
            'change_speed': True,
            'change_route': True,
            'change_curvature': False,
        }

        #self.traffic_dic_path = None
        self.road_dic_path = None

        if self.dataset == 'Waymo':
            # self.relation_gt_path = 'pickles/gt_direct_relation_WOMD_validation_interactive.pickle'
            self.relation_gt_path = 'pickles/gt_direct_relation_WOMD_training_interactive.pickle'
        elif self.dataset == 'NuPlan':
            # self.relation_gt_path = 'pickles/relation_gt_0_5'
            self.relation_gt_path = 'pickles/relation_gt_merged.pickle'
            #self.traffic_dic_path = 'pickles/traffic_dic.pkl'
            self.road_dic_path = 'pickles/road_dic.pkl'

        self.traffic_dic_path = str(Path.home()) + "/nuplan/dataset/pickles/traffic_dic.pkl"
        self.road_dic_path = str(Path.home()) + "/nuplan/dataset/pickles/road_dic.pkl"
        self.relation_gt_path = None
        self.predict_device = 'cpu'  # 'mps' # 'cuda'
        self.use_rl = False
        self.use_bc = True
        # for NSM
        self.nsm = True
        self.nsm_label_path = str(Path.home()) + '/gt_labels/intentions/nuplan_boston/training.wtime.0-100.iter0.pickle'

class EnvConfig(object):
    """
    running mode: 0=debug, 1=test planning algorithm, 2=playback previous results
    overwrite config parameters to change defaults
    """
    env = Config()
    assert env.planning_from >= 10, 'at least reserve 10 frames to measure steady velocity'
    env.predict_relations_for_ego = True
    env.predict_relations_for_env = False
    env.ego_planner = 'dummy_rl'
    env.dynamic_env_planner = 'env_lite'
    env.running_mode = 0


    # env.data_path = {
    #     'NUPLAN_DATA_ROOT': "/public/MARS/datasets/nuPlan",
    #     'NUPLAN_MAPS_ROOT': "/public/MARS/datasets/nuPlan/nuplan-maps-v1.0",
    #     'NUPLAN_DB_FILES': "/public/MARS/datasets/nuPlan/nuplan-v1.0/data/cache/public_set_boston_train",
    # }


class BCRunnerConfig(object):

    def __init__(self):
        self.n_envs = 1
        self.HISTORY_NUM=10
        self.PREDICT_NUM=10

        self.observation_encode_kwargs = dict(
            max_vector_num=1000000,
            max_lane_num=1000,
            max_point_num=2500,
            visible_dis=30,
            max_dis=500,
            scale=1.0,
            stride=5,
            raster_shape=[224, 224, 60],
            raster_scale=1.0,
            history_frame_num=self.HISTORY_NUM,
            predict_length=self.PREDICT_NUM,
            high_res_raster_shape=[224, 224, 35],  # for high resolution image, we cover 50 meters for delicated short-term actions
            high_res_raster_scale=4.0,
            low_res_raster_shape=[224, 224, 35],  # for low resolution image, we cover 300 meters enough for 8 seconds straight line actions
            low_res_raster_scale=0.77,
        )
        
        self.training_kwargs = dict(
            sample_rate=0.25,
            epochs=50,
            max_norm=200,
            savedir="test"
        )
        self.opt_kwargs = dict(
            lr=1e-4, 
            eps=1e-6, 
            weight_decay=0.2,
        )

        self.model_kwargs = dict(
            context_length=self.HISTORY_NUM,
            predict_length=self.PREDICT_NUM,
            method='raster',
            mode="BC",
            loss="mse",
            encoder_kwargs=dict(
                hid_size=128,
                inchannel=60,    
            ),
            ct_kwargs=GPTConfig(
                obs_dim=128, 
                n_embd=128, 
                vocab_size=4,
                block_size=200,
                model_type="naive",
                max_timestep=int(1e5),
                embd_pdrop=0.5,
                resid_pdrop=0.2,
                attn_pdrop=0.2,
                n_layer=5,
                pred_layers=2,
                rtg_layers=2,
                bc_layers=2,
                cont_action=True,
                n_head=1,
               ),
            decoder_kwargs=dict(
                hid_size=128,
                output_size=4,
                init_log_std=0.,
            ),

        ) 

    def _convert2dic(self):
        return dict((name, getattr(self, name)) for name in dir(self) if not name.startswith('_')) 

class RLRunnerConfig(object):
    def __init__(self):
        self.HISTORY_NUM=10

        self.observation_encode_kwargs = dict(
            max_vector_num=1000000,
            max_lane_num=1000,
            max_point_num=2500,
            visible_dis=30,
            max_dis=80,
            scale=1.0,
            stride=5,
            raster_shape=[224, 224, 60],
            raster_scale=1.0,
            history_frame_num=self.HISTORY_NUM,
        )
    
        self.model_kwargs = dict(
            encoder_kwargs=dict(
                hid_size=128,
                inchannel=60,
                history_frame_num=1,
                predict_length=1,
                raster_only=True
            ),
            ct_kwargs=GPTConfig(
                obs_dim=128, 
                n_embd=128, 
                vocab_size=4,
                block_size=20,
                model_type="naive",
                max_timestep=int(1e5),
                embd_pdrop=0.5,
                resid_pdrop=0.2,
                attn_pdrop=0.2,
                n_layer=5,
                pred_layers=2,
                rtg_layers=2,
                bc_layers=2,
                cont_action=True,
                n_head=1,
               ),
            decoder_kwargs=dict(
                hid_size=128,
                output_size=4,
                init_log_std=0.,
            ),
            mode="RL",
            loss="mse",
        ) 

if __name__ == '__main__':
    config = BCRunnerConfig()
    dic = config._convert2dic()
    config.HISTORY_NUM = 88
    print(config.model_kwargs["encoder_kwargs"]["history_frame_num"])
