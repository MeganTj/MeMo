from gym.envs.registration import register

register(
    id = 'RobotLocomotion-v0',
    entry_point = 'environments.robot_locomotion:RobotLocomotionEnv',
    max_episode_steps=128,
)

register(
    id = 'RobotLocomotion-v1',
    entry_point = 'environments.robot_locomotion_full_state:RobotLocomotionFullEnv',
    max_episode_steps=128,
)

register(
    id = 'RobotLocomotion-v2',
    entry_point = 'environments.robot_locomotion_full_state:RobotLocomotionFullEnvV2',
    max_episode_steps=128,
)