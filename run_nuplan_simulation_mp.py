from typing import List

import copy
import functools
import yaml
import datetime
import torch
import argparse
import warnings
import os
from tqdm import tqdm
from nuplan_simulation.planner import Planner
from nuplan_simulation.common_utils import *
import hydra

warnings.filterwarnings("ignore")

from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback
from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.main_callback.metric_summary_callback import MetricSummaryCallback
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.controller.log_playback import LogPlaybackController
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import StepSimulationTimeController
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.nuboard.base.data_class import NuBoardFile
import logging
import numpy as np

logger = logging.getLogger(__name__)

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner


from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample

# multiple proces
import torch.multiprocessing as mp


# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9506))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def deep_update(original, update):
    for key, value in update.items():
        if isinstance(value, dict):
            original[key] = deep_update(original.get(key, {}), value)
        else:
            original[key] = value
    return original

def process_single_sample(i, planners, planner_inputs):
    return planners[i].inputs_to_model_sample(
        history=planner_inputs[i].history,
        traffic_light_data=list(planner_inputs[i].traffic_light_data),
    )


def run_metric_engine(
    metric_engine: MetricsEngine, scenario: AbstractScenario, planner_name: str, history: SimulationHistory
) -> None:
    """
    Run the metric engine.
    """
    logger.debug("Starting metrics computation...")
    # metric_files = metric_engine.compute(history, scenario=scenario, planner_name=planner_name)
    metric_results = metric_engine.compute_metric_results(history, scenario)
    # print('testing: ', metric_results)
    return metric_results


def update_metric_results(metric_dic, batch_metric_results):
    if metric_dic is None:
        metric_dic = {}
        for each_key in batch_metric_results[0]:
            if batch_metric_results[0][each_key][0].metric_score is not None:
                metric_dic[each_key] = []

    for each in batch_metric_results:
        for key in each.keys():
            if key in metric_dic.keys():
                metric_dic[key] += [each[key][0].metric_score]
    return metric_dic

def compute_overall_score(metric_dic, experiment):
    name, metric_weights, file_name, multiple_metrics, challenge_name = get_aggregator_config(experiment)
    overall_score_dic = {}
    total_scenario = 0
    # average scores over all scenarios
    for key in metric_dic.keys():
        overall_score_dic[key] = np.mean(metric_dic[key].copy())
        total_scenario = len(metric_dic[key])
    print('Scores on each key: ', overall_score_dic)
    scenario_scores = []
    for i in range(total_scenario):
        scenario_score = []
        total_weights = []
        multiplier = 1
        for key in metric_dic.keys():
            if key in metric_weights.keys():
                scenario_score.append(metric_dic[key][i] * metric_weights[key])
                total_weights.append(metric_weights[key])
            elif key in multiple_metrics:
                multiplier *= metric_dic[key][i]
            else:
                print('WARNING, unknown key ', key)
                scenario_score.append(metric_dic[key][i])
                total_weights.append(1.0)
        # assert sum(total_weights) == 16, total_weights
        scenario_scores.append(sum(scenario_score) / sum(total_weights) * multiplier)
    print('Score per scenarios: ', scenario_scores)
    return overall_score_dic, np.mean(scenario_scores)

def compute_overall_score_CLS(metric_dic):
    overall_score_dic = {}
    # average scores over all scenarios
    for key in metric_dic.keys():
        overall_score_dic[key] = np.mean(metric_dic[key].copy())
    # score_list = []
    weights = {
        'ego_progress_along_expert_route': 5,
        'time_to_collision_within_bound': 5,
        'speed_limit_compliance': 4,
        'ego_is_comfortable': 2
    }
    multiple_metrics = ['no_ego_at_fault_collisions', 'drivable_area_compliance',
                        'ego_is_making_progress', 'driving_direction_compliance']
    print('Scores on each key: ', overall_score_dic)
    # compute overall score with weights

    total_scenario = len(metric_dic['ego_progress_along_expert_route'])
    scenario_scores = []
    for i in range(total_scenario):
        scenario_score = []
        total_weights = []
        multiplier = 1
        for key in metric_dic.keys():
            if key in weights.keys():
                scenario_score.append(metric_dic[key][i] * weights[key])
                total_weights.append(weights[key])
            elif key in multiple_metrics:
                multiplier *= metric_dic[key][i]
            else:
                print('WARNING, unknown key ', key)
                scenario_score.append(metric_dic[key][i])
                total_weights.append(1.0)
        assert sum(total_weights) == 16, total_weights
        scenario_scores.append(sum(scenario_score) / sum(total_weights) * multiplier)
    print('Score per scenarios: ', scenario_scores)
    # for key in overall_score_dic.keys():
    #     if key in weights.keys():
    #         score_list.append(overall_score_dic[key] * weights[key])
    #         total_weights += weights[key]
    #     else:
    #         score_list.append(overall_score_dic[key])
    #         total_weights += 1
    # return overall_score_dic, float(sum(score_list)) / total_weights
    return overall_score_dic, np.mean(scenario_scores)


def compute_overall_score_OLS(metric_dic):
    overall_score_dic = {}
    # average scores over all scenarios
    for key in metric_dic.keys():
        overall_score_dic[key] = np.mean(metric_dic[key])
    score_list = []
    weights = {
        'planner_expert_average_heading_error_within_bound': 2,
        'planner_expert_final_heading_error_within_bound': 2,
    }
    print('Scores on each key: ', overall_score_dic)
    # compute overall score with weights
    total_weights = 0.0
    for key in overall_score_dic.keys():
        if key in weights.keys():
            score_list.append(overall_score_dic[key] * weights[key])
            total_weights += weights[key]
        else:
            score_list.append(overall_score_dic[key])
            total_weights += 1
    return overall_score_dic, float(sum(score_list)) / total_weights



class SimulationRunnerBatch(SimulationRunner):
    """
    Overwrite the run method for testing large models with batches of scenarios.
    """

    def __init__(self, simulations: Simulation, planners: AbstractPlanner, model=None, args={}):
        """
        Initialize the simulations manager
        :param simulation: Simulation which will be executed
        :param planner: to be used to compute the desired ego's trajectory
        """
        self._simulations = simulations
        self._planners = planners
        self._model = model
        self._batch_size = len(simulations)
        self._args = args

    @property
    def simulations(self):
        return self._simulations

    @property
    def planners(self):
        return self._planners

    def _initialize(self) -> None:
        """
        Initialize the planner
        """
        # Execute specific callback
        for i in range(self._batch_size):
            self._simulations[i].callback.on_initialization_start(self._simulations[i].setup, self.planners[i])

        for i in range(self._batch_size):
            # Initialize Planner
            # Reuse the same model
            self.planners[i].initialize(self._simulations[i].initialize())
            if "str" in self._args.planner_name:
                self.planners[i]._str_generator._model = self._model

        # Execute specific callback
        for i in range(self._batch_size):
            self._simulations[i].callback.on_initialization_end(self._simulations[i].setup, self.planners[i])

    def run(self) -> List[RunnerReport]:
        """
        Run through all simulations. The steps of execution follow:
         - Initialize all planners
         - Step through simulations until there no running simulation
        :return: List of SimulationReports containing the results of each simulation
        """
        start_time = time.perf_counter()

        debug = False
        try:
            debug = args.debug
        except:
            pass

        # Initialize reports for all the simulations that will run
        reports = [RunnerReport(
            succeeded=True,
            error_message=None,
            start_time=start_time,
            end_time=None,
            planner_report=None,
            scenario_name=self._simulations[i].scenario.scenario_name,
            planner_name=self.planners[i].name(),
            log_name=self._simulations[i].scenario.log_name,
        ) for i in range(self._batch_size)]

        # Execute specific callback
        for i in range(self._batch_size):
            self._simulations[i].callback.on_simulation_start(self._simulations[i].setup)

        # Initialize all simulations
        self._initialize()
        simulation_running = True

        step = 0
        while simulation_running:
            # print("step: ", step)
            for i in range(self._batch_size):
                self._simulations[i].callback.on_step_start(self._simulations[i].setup, self.planners[i])

            # print("planning: ", step)
            # Perform step
            planner_inputs = []
            for i in range(self._batch_size):
                planner_input = self._simulations[i].get_planner_input()
                planner_inputs.append(planner_input)

            # Execute specific callback
            for i in tqdm(range(self._batch_size), desc='Step planner starting in batch', disable=not debug):
                self._simulations[i].callback.on_planner_start(self._simulations[i].setup, self.planners[i])
            model_sample_start_time = time.perf_counter()

            trajectories = []
            if "str" in self._args.planner_name:
                model_samples = []
                for i in tqdm(range(self._batch_size), desc='Step model inputs converting in batch', disable=not debug):
                    model_samples.append(self.planners[i]._str_generator._inputs_to_model_sample(
                        history=planner_inputs[i].history,
                        traffic_light_data=list(planner_inputs[i].traffic_light_data),
                        map_name=self.planners[i]._map_api.map_name,
                    ))
                map_names = [self.planners[i]._map_api.map_name for i in range(self._batch_size)]
                # print(f'\nModel sample time: {time.perf_counter() - model_sample_start_time:.3f} s')
                # pack in batch
                samples_in_batch = {key: np.stack([sample[key] for sample in model_samples]) for key in model_samples[0].keys()}
                # model forward in batch
                states_in_batch, scores_in_batch = self.planners[0]._str_generator.predict_states_in_batch(
                    planner_inputs=planner_inputs, 
                    map_names=map_names, 
                    samples_in_batch=samples_in_batch,
                    ego_states_in_batch=[planner_inputs[i].history.ego_states for i in range(self._batch_size)],
                    route_ids=[self.planners[i]._str_generator._route_roadblock_ids for i in range(self._batch_size)],
                    road_dics=[self.planners[i]._str_generator.road_dic for i in range(self._batch_size)],
                )

                planner_start_time = time.perf_counter()
                # Plan path based on all planner's inputs
                for planner, current_input, str_pred_states in zip(self.planners, planner_inputs, states_in_batch):
                    trajectories.append(planner.compute_planner_trajectory(current_input, str_pred_states))
            else:
                for planner, current_input in zip(self.planners, planner_inputs):
                    trajectories.append(planner.compute_planner_trajectory(current_input))
            # trajectories = self.planners[0].compute_planner_trajectory_in_batch(
            #     model_samples=samples_in_batch,
            #     map_names=[self.planners[i]._map_api.map_name for i in range(self._batch_size)],
            #     ego_states_in_batch=[planner_inputs[i].history.ego_states for i in range(self._batch_size)],
            #     route_ids=[self.planners[i]._route_roadblock_ids for i in range(self._batch_size)],
            #     road_dics=[self.planners[i].road_dic for i in range(self._batch_size)],
            # )
            # for i in range(self._batch_size):
            #     if i == 0:
            #         continue
            #     self.planners[i].iteration += 1

            # print(f'Step {planner_inputs[0].iteration.index + 1} with timestamp {planner_inputs[0].iteration.time_s} Planning time: {time.perf_counter() - planner_start_time:.3f} s')
            # print("running: ", step)
            for i in range(self._batch_size):
                # Propagate simulation based on planner trajectory
                self._simulations[i].callback.on_planner_end(self._simulations[i].setup, self.planners[i], trajectories[i])
                self._simulations[i].propagate(trajectories[i])

                # Execute specific callback
                self.simulations[i].callback.on_step_end(self.simulations[i].setup, self.planners[i], self.simulations[i].history.last())

                # Store reports for simulations which just finished running
                current_time = time.perf_counter()
                if not self.simulations[i].is_simulation_running():
                    reports[i].end_time = current_time

                # end loop if any simulation in the batch ends
                simulation_running = simulation_running and self._simulations[i].is_simulation_running()

            # print("finish step: ", step)
            # print("============================")
            if "str" in self._args.planner_name and self._model.config.simulate_one_step_on_training and self._model.training and step > 5:
                # simulate one step on training
                simulation_running = False
            else:
                step += 1

        print("finish RunnerBatch")
        # Delete model to avoid crashes while saving the planner
        for i in range(self._batch_size):
            self.planners[i]._str_generator._model = None
            # Execute specific callback
            self.simulations[i].callback.on_simulation_end(self.simulations[i].setup, self.planners[i], self.simulations[i].history)
            planner_report = self.planners[i].generate_planner_report()
            reports[i].planner_report = planner_report

        print("finish reports")
        return reports


def build_simulation_experiment_folder(output_dir, simulation_dir, metric_dir, aggregator_metric_dir):
    """
    Builds the main experiment folder for simulation.
    :return: The main experiment folder path.
    """
    # print('Building experiment folders...')

    exp_folder = pathlib.Path(output_dir)
    print(f'\nFolder where all results are stored: {exp_folder}\n')
    exp_folder.mkdir(parents=True, exist_ok=True)

    # Build nuboard event file.
    nuboard_filename = exp_folder / (f'nuboard_{int(time.time())}' + NuBoardFile.extension())
    nuboard_file = NuBoardFile(
        simulation_main_path=str(exp_folder),
        simulation_folder=simulation_dir,
        metric_main_path=str(exp_folder),
        metric_folder=metric_dir,
        aggregator_metric_folder=aggregator_metric_dir,
    )

    metric_main_path = exp_folder / metric_dir
    metric_main_path.mkdir(parents=True, exist_ok=True)

    nuboard_file.save_nuboard_file(nuboard_filename)
    # print('Building experiment folders...DONE!')

    return exp_folder.name


def build_simulation(experiment, planner, scenarios, output_dir, simulation_dir, metric_dir, save_reports=True):
    runner_reports = []
    print(f'Building simulations from {len(scenarios)} scenarios...')

    metric_engine = build_metrics_engine(experiment, output_dir, metric_dir)
    print('Building metric engines...DONE\n')

    # Iterate through scenarios
    for scenario in tqdm(scenarios, desc='Running simulation', disable=not args.debug):
        # Ego Controller and Perception
        if experiment == 'open_loop_boxes':
            ego_controller = LogPlaybackController(scenario)
            observations = TracksObservation(scenario)
        elif experiment == 'closed_loop_nonreactive_agents':
            tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0],
                                 r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10,
                                 jerk_penalty=1e-4, curvature_rate_penalty=1e-2,
                                 stopping_proportional_gain=0.5, stopping_velocity=0.2)
            motion_model = KinematicBicycleModel(get_pacifica_parameters())
            ego_controller = TwoStageController(scenario, tracker, motion_model)
            observations = TracksObservation(scenario)
        elif experiment == 'closed_loop_reactive_agents':
            tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0],
                                 r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10,
                                 jerk_penalty=1e-4, curvature_rate_penalty=1e-2,
                                 stopping_proportional_gain=0.5, stopping_velocity=0.2)
            motion_model = KinematicBicycleModel(get_pacifica_parameters())
            ego_controller = TwoStageController(scenario, tracker, motion_model)
            observations = IDMAgents(target_velocity=10, min_gap_to_lead_agent=1.0, headway_time=1.5,
                                     accel_max=1.0, decel_max=2.0, scenario=scenario,
                                     open_loop_detections_types=["PEDESTRIAN", "BARRIER", "CZONE_SIGN", "TRAFFIC_CONE",
                                                                 "GENERIC_OBJECT"])
        else:
            raise ValueError(f"Invalid experiment type: {experiment}")

        print('Running on scenario: ', scenario.token, scenario.get_number_of_iterations())

        # Simulation Manager
        simulation_time_controller = StepSimulationTimeController(scenario)

        # Stateful callbacks
        metric_callback = MetricCallback(metric_engine=metric_engine)
        # sim_log_callback = SimulationLogCallback(output_dir, simulation_dir, "msgpack")

        # Construct simulation and manager
        simulation_setup = SimulationSetup(
            time_controller=simulation_time_controller,
            observations=observations,
            ego_controller=ego_controller,
            scenario=scenario,
        )

        simulation = Simulation(
            simulation_setup=simulation_setup,
            callback=MultiCallback([metric_callback])
            # callback=MultiCallback([metric_callback, sim_log_callback])
        )

        # Begin simulation
        simulation_runner = SimulationRunner(simulation, planner)
        report = simulation_runner.run()
        runner_reports.append(report)

    # save reports
    save_runner_reports(runner_reports, output_dir, 'runner_reports')

    # Notify user about the result of simulations
    failed_simulations = str()
    number_of_successful = 0

    for result in runner_reports:
        if result.succeeded:
            number_of_successful += 1
        else:
            print("Failed Simulation.\n '%s'", result.error_message)
            failed_simulations += f"[{result.log_name}, {result.scenario_name}] \n"

    number_of_failures = len(scenarios) - number_of_successful
    print(f"Number of successful simulations: {number_of_successful}")
    print(f"Number of failed simulations: {number_of_failures}")

    # Print out all failed simulation unique identifier
    if number_of_failures > 0:
        print(f"Failed simulations [log, token]:\n{failed_simulations}")

    print('Finished running simulations!')

    return runner_reports


def build_simulation_in_batch_multiprocess(experiment, scenarios, output_dir, simulation_dir, metric_dir, batch_size=32, model=None, args=None,
                              all_road_dic={}, save_reports=True, controller='two_stage_controller'):
    runner_reports = []
    # print(f'Building simulations from {len(scenarios)} scenarios...')
    metric_engine = build_metrics_engine(experiment, output_dir, metric_dir)
    # print('Building metric engines...DONE\n')

    N_scenarios = len(scenarios)
    split_size = int(np.ceil(N_scenarios / args.node_num))
    print(f"total scenarios: {N_scenarios}")
    
    begin_id = args.local_rank * split_size
    end_id = min((args.local_rank + 1) * split_size, N_scenarios)
    print(f"locally deal with split: {[begin_id, end_id]}")
    
    scenarios = scenarios[begin_id: end_id]
    N_scenarios = len(scenarios)
    print(f"local scenarios: {N_scenarios}")
    
    if len(scenarios) % batch_size > 0:
        aux_scenarios = [scenarios[-1] for i in range(batch_size - len(scenarios) % batch_size)]
        scenarios += aux_scenarios
        print(f"created {len(aux_scenarios)} auxiliary scenarios to make sure N_scenarios % bs = 0")
    
    # if len(scenarios) % batch_size > 0:
    #     print(f"Batch size {batch_size} is not a divisor of the number of scenarios {len(scenarios)}, skipping the last batch.")
    #     scenarios = scenarios[:-(len(scenarios) % batch_size)]

    scenario_groups = [scenarios[i:i + batch_size] for i in range(0, len(scenarios), batch_size)]
    over_all_metric_results = None

    assert len(scenario_groups) > 0, f'no scenarios after filtering, check folder and yaml file.'
    
    # multi-processing
    rep_cnt = args.processes_repetition
    gpu_cnt = torch.cuda.device_count()
    process_cnt = gpu_cnt * rep_cnt

    # processes resources
    tasks_queue = mp.Queue()
    results_queue = mp.Queue()
    locks = [mp.Manager().Lock() for _ in range(gpu_cnt)]

    # global model
    # model_planner = Planner(args.model_path, 'cpu')
    # model_planner._initialize_model()
    # global_model = model_planner._model
    params_planner = yaml.safe_load(open(f"{os.getcwd()}/tuplan_garage/planning/script/config/simulation/planner/{args.planner_name}.yaml", 'r'))
    if "str" in args.planner_name:
        deep_update(params_planner, args.planner_override)
    model_planner = hydra.utils.instantiate(params_planner)[args.planner_name]
    if "str" in args.planner_name:
        global_model = model_planner._str_generator._model
        _time = time.time()
        global_model_cuda = [copy.deepcopy(global_model).to(f'cuda:{i}') for i in range(gpu_cnt)]
        global_model_with_lock =  [_inject_model(model, lock) for model, lock in zip(global_model_cuda, locks)]
        print(f'Loading model time: {time.time()-_time:.2f} s.')
    else:
        global_model_with_lock = [None] * gpu_cnt
    

    # process init
    processes = []
    for i in tqdm(list(range(process_cnt)), desc='Starting Multi-Processes'):
        gpu_idx = i % gpu_cnt
        p = mp.Process(target=Worker(global_model_with_lock[gpu_idx], gpu_idx, locks[gpu_idx], args), args=(tasks_queue, results_queue))
        p.start()
        processes.append(p)

    print('start putting tasks to queue')

    # Iterate through scenarios
    N = len(scenario_groups)
    for scenarios_onegroup in scenario_groups:
        tasks_queue.put((scenarios_onegroup, all_road_dic, batch_size, experiment, controller, output_dir, simulation_dir, metric_engine))
    for _ in range(process_cnt):
        tasks_queue.put(None)
    
    print('start fetching results from queue')

    # fetch results
    failed_scenarios = []
    for _ in tqdm(list(range(N)), desc='Fetch Results'):
        is_succ, ret_msg = results_queue.get()

        if not is_succ:
            error_str = (f'Exception caught in scenario: {_}\n'
                         f'Error message: {ret_msg}')
            print(error_str)
            failed_scenarios.append(error_str)
            continue

        batch_metric_results, reports = ret_msg
        over_all_metric_results = update_metric_results(
            metric_dic=over_all_metric_results,
            batch_metric_results=batch_metric_results
        )
        runner_reports += reports

    if failed_scenarios:
        err_file = 'err_scenarios.log'
        print(f'{len(failed_scenarios)} scenarios failed during simulation.\n'
              f'writting to {err_file}.')
        with open(err_file, 'w') as f:
            for err_msg in failed_scenarios:
                f.write(f"{err_msg}\n\n")

    # finish
    [p.join() for p in processes]

    # compute overall score
    overall_score_dic, overall_score = compute_overall_score(over_all_metric_results, experiment)
    print(f'Overall score: {overall_score}')

    # save reports
    if save_reports:
        save_runner_reports(runner_reports, output_dir, 'runner_reports')

    # Notify user about the result of simulations
    failed_simulations = str()
    number_of_successful = 0

    for result in runner_reports:
        if result.succeeded:
            number_of_successful += 1
        else:
            print("Failed Simulation.\n '%s'", result.error_message)
            failed_simulations += f"[{result.log_name}, {result.scenario_name}] \n"

    number_of_failures = N_scenarios - number_of_successful
    print(f"Number of successful simulations: {number_of_successful}")
    print(f"Number of failed simulations: {number_of_failures}")

    # Print out all failed simulation unique identifier
    if number_of_failures > 0:
        print(f"Failed simulations [log, token]:\n{failed_simulations}")

    print('Finished running simulations!')

    return runner_reports, overall_score_dic, overall_score


class Worker():
    def __init__(self, model, gpu_idx, lock, args):
        self.model = model
        self.args = args
    
    def __call__(self, tasks_queue, results_queue):
        while True:
            task = tasks_queue.get()
            if task is None:
                break
            try:
                result = (True, _worker_func(*task, self.model, self.args))
            except Exception as e:
                result = (False, str(e))
            results_queue.put(result)

def _generate(self, func, lock, *args, **kwargs):
    lock.acquire()

    try:
        return func(*args, **kwargs)
    except Exception as e:
        raise e
    finally:
        lock.release()

def _inject_model(model, lock):
    model.generate = functools.partial(_generate, model, model.generate, lock)
    return model


def _worker_func(scenarios, all_road_dic, batch_size, experiment, controller, output_dir, simulation_dir, metric_engine, global_model, args):
    # running each batch of scenarios
    # Initialize new planners for each scenario
    
    params_planner = yaml.safe_load(open(f"{os.getcwd()}/tuplan_garage/planning/script/config/simulation/planner/{args.planner_name}.yaml", 'r'))
    if "str" in args.planner_name:
        deep_update(params_planner, args.planner_override)
        params_planner['str_closed_planner']['str_generator']['build_model'] = False
    # model_planner = hydra.utils.instantiate(params_planner)[args.planner_name]
    
    planners  = [hydra.utils.instantiate(params_planner)[args.planner_name] for _ in range(batch_size)]
    
    # planners = [Planner(model_path=None, device=None, all_road_dic=all_road_dic, scenarios=scenarios) for _ in range(batch_size)]
    # Initialize Ego Controller and Perception
    if experiment == 'open_loop_boxes':
        ego_controllers = [LogPlaybackController(scenario) for scenario in scenarios]
        observations = [TracksObservation(scenario) for scenario in scenarios]
    elif experiment == 'closed_loop_nonreactive_agents':
        if controller == 'perfect_controller':
            ego_controllers = [PerfectTrackingController(scenario) for scenario in scenarios]
        else:
            tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0],
                                    r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10,
                                    jerk_penalty=1e-4, curvature_rate_penalty=1e-2,
                                    stopping_proportional_gain=0.5, stopping_velocity=0.2)
            motion_model = KinematicBicycleModel(get_pacifica_parameters())
            ego_controllers = [TwoStageController(scenario, tracker, motion_model) for scenario in scenarios]
        observations = [TracksObservation(scenario) for scenario in scenarios]
    elif experiment == 'closed_loop_reactive_agents':
        if controller == 'perfect_controller':
            ego_controllers = [PerfectTrackingController(scenario) for scenario in scenarios]
        else:
            tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0],
                                    r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10,
                                    jerk_penalty=1e-4, curvature_rate_penalty=1e-2,
                                    stopping_proportional_gain=0.5, stopping_velocity=0.2)
            motion_model = KinematicBicycleModel(get_pacifica_parameters())
            ego_controllers = [TwoStageController(scenario, tracker, motion_model) for scenario in scenarios]
        observations = [IDMAgents(target_velocity=10, min_gap_to_lead_agent=1.0, headway_time=1.5,
                                    accel_max=1.0, decel_max=2.0, scenario=scenario,
                                    open_loop_detections_types=["PEDESTRIAN", "BARRIER", "CZONE_SIGN", "TRAFFIC_CONE",
                                                                "GENERIC_OBJECT"]) for scenario in scenarios]
    else:
        raise ValueError(f"Invalid experiment type: {experiment}")

    # print('Running on new batch scenario: ', len(scenarios), scenarios[0].token, scenarios[0].get_number_of_iterations())

    # Simulation Manager
    simulation_time_controllers = [StepSimulationTimeController(scenario) for scenario in scenarios]

    # Stateful callbacks
    metric_callbacks = [MetricCallback(metric_engine=metric_engine) for _ in range(batch_size)]
    sim_log_callbacks = [SimulationLogCallback(output_dir, simulation_dir, "msgpack") for _ in range(batch_size)]

    # Construct simulation and manager
    simulation_setups = [SimulationSetup(
        time_controller=simulation_time_controller,
        observations=observation,
        ego_controller=ego_controller,
        scenario=scenario,
    ) for simulation_time_controller, observation, ego_controller, scenario in zip(simulation_time_controllers, observations, ego_controllers, scenarios)
    ]

    simulations = [Simulation(
        simulation_setup=simulation_setup,
        callback=MultiCallback([metric_callback])
        # callback=MultiCallback([metric_callback, sim_log_callback])
    ) for simulation_setup, metric_callback, sim_log_callback in zip(simulation_setups, metric_callbacks, sim_log_callbacks)
    ]

    # Begin simulation
    simulation_runner = SimulationRunnerBatch(simulations, planners, global_model, args)
    reports_tmp = simulation_runner.run()

    # remove repeated reports
    reports = []
    reports_names = set()
    for report in reports_tmp:
        if report.scenario_name not in reports_names:
            reports_names.add(report.scenario_name)
            reports.append(report)

    # inspect reports
    batch_metric_results = []
    for i in range(len(reports)):
        metric_callback = simulations[i].callback.callbacks[0]
        metric_engine = metric_callback.metric_engine
        metric_results = run_metric_engine(metric_engine, simulations[i].scenario, planners[i].name(), simulations[i].history)
        batch_metric_results.append(metric_results)
    
    return batch_metric_results, reports


def build_nuboard(scenario_builder, simulation_path):
    nuboard = NuBoard(
        nuboard_paths=simulation_path,
        scenario_builder=scenario_builder,
        vehicle_parameters=get_pacifica_parameters(),
        port_number=5006
    )

    nuboard.run()


def main(args):
    # parameters
    experiment_name = args.test_type  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    job_name = args.planner_name
    experiment_time = args.exp_folder if args.exp_folder is not None else datetime.datetime.now()
    experiment = f"{experiment_name}/{job_name}/{experiment_time}"
    output_dir = f"testing_log/{experiment}"
    simulation_dir = "simulation"
    metric_dir = "metrics"
    aggregator_metric_dir = "aggregator_metric"

    # initialize planner
    torch.set_grad_enabled(False)

    # initialize main aggregator
    metric_aggregators = build_metrics_aggregators(experiment_name, output_dir, aggregator_metric_dir)
    metric_save_path = f"{output_dir}/{metric_dir}"
    metric_aggregator_callback = MetricAggregatorCallback(metric_save_path, metric_aggregators)
    metric_file_callback = MetricFileCallback(metric_file_output_path=f"{output_dir}/{metric_dir}",
                                              scenario_metric_paths=[f"{output_dir}/{metric_dir}"],
                                              delete_scenario_metric_files=True)
    metric_summary_callback = MetricSummaryCallback(metric_save_path=f"{output_dir}/{metric_dir}",
                                                    metric_aggregator_save_path=f"{output_dir}/{aggregator_metric_dir}",
                                                    summary_output_path=f"{output_dir}/summary",
                                                    num_bins=20, pdf_file_name='summary.pdf')
    main_callbacks = MultiMainCallback([metric_file_callback, metric_aggregator_callback, metric_summary_callback])
    main_callbacks.on_run_simulation_start()

    # build simulation folder
    build_simulation_experiment_folder(output_dir, simulation_dir, metric_dir, aggregator_metric_dir)

    # build scenarios
    print('Extracting scenarios...', args.load_without_yaml)
    map_version = "nuplan-maps-v1.0"
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, None, map_version, scenario_mapping=scenario_mapping)
    if not args.load_without_yaml:
        print('Filtering with yaml file...')
        params = yaml.safe_load(open(args.split_filter_yaml, 'r'))
        params['log_names'] = None
        scenario_filter = hydra.utils.instantiate(params)
        # scenario_filter = ScenarioFilter(**params)
    else:
        print('Filtering with types ...')
        scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type))
    worker = SingleMachineParallelExecutor(use_process_pool=False)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print('Got scenarios ', len(scenarios))
    if args.max_scenario_num > 0:
        scenarios = scenarios[:args.max_scenario_num]

    if args.board_only_log_path is not None:
        output_dir = args.board_only_log_path
        simulation_file = [str(file) for file in pathlib.Path(output_dir).iterdir() if
                           file.is_file() and file.suffix == '.nuboard']

        # show metrics and scenarios
        if args.show_nuboard:
            build_nuboard(builder, simulation_file)
        return

    # begin testing
    print('Running simulations...')
    if args.batch_size < 1:
        params_planner = yaml.safe_load(open(f"{os.getcwd()}/tuplan_garage/planning/script/config/simulation/planner/{args.planner_name}.yaml", 'r'))
        if "str" in args.planner_name:
            deep_update(params_planner, args.planner_override)
        planner = hydra.utils.instantiate(params_planner)[args.planner_name]
        build_simulation(experiment_name, planner, scenarios, output_dir, simulation_dir, metric_dir)
    else:
        build_simulation_in_batch_multiprocess(experiment_name, scenarios, output_dir, simulation_dir, metric_dir, args.batch_size, args=args)
    main_callbacks.on_run_simulation_end()
    simulation_file = [str(file) for file in pathlib.Path(output_dir).iterdir() if
                       file.is_file() and file.suffix == '.nuboard']

    # show metrics and scenarios
    if args.show_nuboard:
        build_nuboard(builder, simulation_file)

    # use ssh with port forwarding to view the nuboard
    # example: ssh -p 3022* -L 5006:localhost:5006 root@10.210.22.11*


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--map_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--exp_folder', type=str, default=None)
    parser.add_argument('--test_type', type=str, default='closed_loop_nonreactive_agents')
    parser.add_argument('--load_without_yaml', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--scenarios_per_type', type=int, default=20)
    parser.add_argument('--nuplan_exp_root', type=str, default='/cephfs/sunq/nuplan/dataset')
    parser.add_argument('--split_filter_yaml', type=str, default='/cephfs/sunq/StateTransformer/nuplan_simulation/test_split.yaml')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--max_scenario_num', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--board_only_log_path', type=str, default=None)
    parser.add_argument('--processes-repetition', type=int, default=1)
    parser.add_argument('--planner_name', type=str, default="str_closed_planner")
    parser.add_argument('--show_nuboard', action='store_true', default=False)
    parser.add_argument('--node_num', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--pdm_lateral_offsets', type=str, default="none")      # 若要设置，形式例如 "-1 1"
    parser.add_argument('--pdm_speed_limit_fraction', type=str, default="0.2 0.4 0.6 0.8 1.0")
    parser.add_argument('--conservative_factor', type=float, default=1.0)
    parser.add_argument('--comfort_weight', type=float, default=2.0)
    parser.add_argument('--initstable_time', type=int, default=5)
    
    # param of reuse pdm centerline
    def set_nested(dic, keys, value):
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value

    planner_override = dict()
    if parser.parse_args().pdm_lateral_offsets != "none":
        set_nested(planner_override, ['str_closed_planner', 'lateral_offsets'], [float(c) for c in parser.parse_args().pdm_lateral_offsets.split(' ')])
    set_nested(planner_override, ['str_closed_planner', 'idm_policies', 'speed_limit_fraction'], [float(c) for c in parser.parse_args().pdm_speed_limit_fraction.split(' ')])
    set_nested(planner_override, ['str_closed_planner', 'str_generator', 'model_path'], parser.parse_args().model_path)
    set_nested(planner_override, ['str_closed_planner', 'conservative_factor'], parser.parse_args().conservative_factor)
    set_nested(planner_override, ['str_closed_planner', 'comfort_weight'], parser.parse_args().comfort_weight)
    set_nested(planner_override, ['str_closed_planner', 'initstable_time'], parser.parse_args().initstable_time)

    
        
    parser.add_argument('--planner_override', type=dict, default=planner_override)
    args = parser.parse_args()
    os.environ['NUPLAN_EXP_ROOT'] = args.nuplan_exp_root

    main(args)
