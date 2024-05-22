from gym.envs.registration import register

register(
    id = 'AllRelClawGraspSphere-v0',
    entry_point = 'environments.grasping:AllRelClawGraspSphere',
    max_episode_steps=50,
)

register(
    id = 'AllRelClawGraspCube-v0',
    entry_point = 'environments.grasping:AllRelClawGraspCube',
    max_episode_steps=50,
)