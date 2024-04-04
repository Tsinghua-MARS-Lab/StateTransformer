import tensorflow as tf
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics
from waymo_open_dataset.utils import trajectory_utils
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2

from waymo_open_dataset.utils.sim_agents import submission_specs
config = metrics.load_metrics_config()

VALIDATION_FILES = "/public/MARS/datasets/waymo_prediction_v1.2.0/scenario/validation/validation.tfrecord*"

# Define the dataset from the TFRecords.
filenames = tf.io.matching_files(VALIDATION_FILES)
dataset = tf.data.TFRecordDataset(filenames)

def simulate_with_extrapolation(
scenario: scenario_pb2.Scenario,
print_verbose_comments: bool = True) -> tf.Tensor:
    vprint = print if print_verbose_comments else lambda arg: None

    # To load the data, we create a simple tensorized version of the object tracks.
    logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario)
    # Using `ObjectTrajectories` we can select just the objects that we need to
    # simulate and remove the "future" part of the Scenario.
    vprint(f'Original shape of tensors inside trajectories: {logged_trajectories.valid.shape} (n_objects, n_steps)')
    logged_trajectories = logged_trajectories.gather_objects_by_id(
        tf.convert_to_tensor(submission_specs.get_sim_agent_ids(scenario)))
    logged_trajectories = logged_trajectories.slice_time(
        start_index=0, end_index=submission_specs.CURRENT_TIME_INDEX + 1)
    vprint(f'Modified shape of tensors inside trajectories: {logged_trajectories.valid.shape} (n_objects, n_steps)')

    # We can verify that all of these objects are valid at the last step.
    vprint(f'Are all agents valid: {tf.reduce_all(logged_trajectories.valid[:, -1]).numpy()}')

    # We extract the speed of the sim agents (in the x/y/z components) ready for
    # extrapolation (this will be our policy).
    states = tf.stack([logged_trajectories.x, logged_trajectories.y,
                        logged_trajectories.z, logged_trajectories.heading],
                        axis=-1)
    n_objects, n_steps, _ = states.shape
    last_velocities = states[:, -1, :3] - states[:, -2, :3]
    # We also make the heading constant, so concatenate 0. as angular speed.
    last_velocities = tf.concat(
        [last_velocities, tf.zeros((n_objects, 1))], axis=-1)
    # It can happen that the second to last state of these sim agents might be
    # invalid, so we will set a zero speed for them.
    vprint(f'Is any 2nd to last state invalid: {tf.reduce_any(tf.logical_not(logged_trajectories.valid[:, -2]))}')
    vprint(f'This will result in either min or max speed to be really large: {tf.reduce_max(tf.abs(last_velocities))}')
    valid_diff = tf.logical_and(logged_trajectories.valid[:, -1],
                                logged_trajectories.valid[:, -2])
    # `last_velocities` shape: (n_objects, 4).
    last_velocities = tf.where(valid_diff[:, tf.newaxis],
                                last_velocities,
                                tf.zeros_like(last_velocities))
    vprint(f'Now this should be back to a normal value: {tf.reduce_max(tf.abs(last_velocities))}')

    # Now we carry over a simulation. As we discussed, we actually want 32 parallel
    # simulations, so we make this batched from the very beginning. We add some
    # random noise on top of our actions to make sure the behaviours are different.
    # To properly scale the noise, we get the max velocities (average over all
    # objects, corresponding to axis 0) in each of the dimensions (x/y/z/heading).
    NOISE_SCALE = 0.01
    # `max_action` shape: (4,).
    max_action = tf.reduce_max(last_velocities, axis=0)
    # We create `simulated_states` with shape (n_rollouts, n_objects, n_steps, 4).
    simulated_states = tf.tile(states[tf.newaxis, :, -1:, :], [submission_specs.N_ROLLOUTS, 1, 1, 1])
    vprint(f'Shape: {simulated_states.shape}')

    for step in range(submission_specs.N_SIMULATION_STEPS):
        current_state = simulated_states[:, :, -1, :]
        # Random actions, take a normal and normalize by min/max actions
        action_noise = tf.random.normal(
            current_state.shape, mean=0.0, stddev=NOISE_SCALE)
        actions_with_noise = last_velocities[None, :, :] + (action_noise * max_action)
        next_state = current_state + actions_with_noise
        simulated_states = tf.concat(
            [simulated_states, next_state[:, :, None, :]], axis=2)

    # We also need to remove the first time step from `simulated_states` (it was
    # still history).
    # `simulated_states` shape before: (n_rollouts, n_objects, 81, 4).
    # `simulated_states`: (n_rollouts, n_objects, 80, 4).
    simulated_states = simulated_states[:, :, 1:, :]
    vprint(f'Final simulated states shape: {simulated_states.shape}')

    return logged_trajectories, simulated_states

# Since these are raw Scenario protos, we need to parse them in eager mode.
dataset_iterator = dataset.as_numpy_iterator()
bytes_example = next(dataset_iterator)
scenario = scenario_pb2.Scenario.FromString(bytes_example)
print(f'Checking type: {type(scenario)}')
logged_trajectories, simulated_states = simulate_with_extrapolation(
    scenario, print_verbose_comments=True)
                                                                                                    
def joint_scene_from_states(
    states: tf.Tensor, object_ids: tf.Tensor
    ) -> sim_agents_submission_pb2.JointScene:
# States shape: (num_objects, num_steps, 4).
# Objects IDs shape: (num_objects,).
    states = states.numpy()
    simulated_trajectories = []
    for i_object in range(len(object_ids)):
        simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
            center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
            center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
            object_id=object_ids[i_object]
        ))
    return sim_agents_submission_pb2.JointScene(
        simulated_trajectories=simulated_trajectories)

# Package the first simulation into a `JointScene`
joint_scene = joint_scene_from_states(simulated_states[0, :, :, :],
                                    logged_trajectories.object_id)
# Validate the joint scene. Should raise an exception if it's invalid.
submission_specs.validate_joint_scene(joint_scene, scenario)

# Now we can replicate this strategy to export all the parallel simulations.
def scenario_rollouts_from_states(
    scenario: scenario_pb2.Scenario,
    states: tf.Tensor, object_ids: tf.Tensor
    ) -> sim_agents_submission_pb2.ScenarioRollouts:
    # States shape: (num_rollouts, num_objects, num_steps, 4).
    # Objects IDs shape: (num_objects,).
    joint_scenes = []
    for i_rollout in range(states.shape[0]):
        joint_scenes.append(joint_scene_from_states(states[i_rollout], object_ids))
    return sim_agents_submission_pb2.ScenarioRollouts(
        # Note: remember to include the Scenario ID in the proto message.
        joint_scenes=joint_scenes, scenario_id=scenario.scenario_id)

scenario_rollouts = scenario_rollouts_from_states(
    scenario, simulated_states, logged_trajectories.object_id)

scenario_metrics = metrics.compute_scenario_metrics_for_bundle(
    config, scenario, scenario_rollouts)
print(scenario_metrics)
exit()