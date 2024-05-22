import sys, os
design_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RoboGrammar', 'examples', 'design_search')
sys.path.append(design_dir)

from copy import deepcopy
from collections import defaultdict
import numpy as np
import pdb

from design_search import presimulate, simulate, build_normalized_robot, make_initial_graph
import pyrobotdesign as rd
import heapq

def make_sim_fn(task, robot, robot_init_pos):
    sim = rd.BulletSimulation(task.time_step)
    task.add_terrain(sim)
    # Rotate 180 degrees around the y axis, so the base points to the right
    sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    return sim

def finalize_robot(robot):
    for link in robot.links:
        link.label = ""
        link.joint_label = ""
        if link.shape == rd.LinkShape.NONE:
            link.shape = rd.LinkShape.CAPSULE
            link.length = 0.1
            link.radius = 0.025
            link.color = [1.0, 0.0, 1.0]
        if link.joint_type == rd.JointType.NONE:
            link.joint_type = rd.JointType.FIXED
            link.joint_color = [1.0, 0.0, 1.0]

def build_robot(args):
    graphs = rd.load_graphs(args.grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]
    rule_sequence = [int(s) for s in args.rule_sequence.split(", ")]

    graph = make_initial_graph()
    for r in rule_sequence:
        matches = rd.find_matches(rules[r].lhs, graph)
        if matches:
            graph = rd.apply_rule(rules[r], graph, matches[0])
    robot = build_normalized_robot(graph)
    finalize_robot(robot)
    # Give the joints the labels which correspond to the action vector
    j = 0
    for i, link in enumerate(robot.links):
        if link.joint_type == rd.JointType.HINGE:
            link.joint_label += str(j)
            j += 1
        else:
            link.joint_label = ""
    return robot

def get_robot_state(sim, robot_id):
    base_tf = np.zeros((4, 4), order = 'f')
    sim.get_link_transform(robot_id, 0, base_tf)
    base_R = deepcopy(base_tf[0:3, 0:3])
    base_pos = deepcopy(base_tf[0:3, 3])

    # anguler velocity first and linear velocity next
    base_vel = np.zeros(6, order = 'f')
    sim.get_link_velocity(robot_id, 0, base_vel)
    
    n_dofs = sim.get_robot_dof_count(robot_id)
    
    joint_pos = np.zeros(n_dofs, order = 'f')
    sim.get_joint_positions(robot_id, joint_pos)
    
    joint_vel = np.zeros(n_dofs, order = 'f')
    sim.get_joint_velocities(robot_id, joint_vel)
    
    state = np.hstack((base_R.flatten(), base_pos, base_vel, joint_pos, joint_vel))

    return state

# Get the robot state with the link transforms for all 
def get_full_robot_state(sim, robot_id):
    orig_state = get_robot_state(sim, robot_id)
    base_pos = orig_state[9:12]
    n_dofs = sim.get_robot_dof_count(robot_id)

    # Iterate through each hinge joint and get their
    # corresponding link's rotation and position
    all_joint_R_pos = []
    for i in range(n_dofs):
        joint_tf = np.zeros((4, 4), order = 'f')
        sim.get_link_transform(robot_id, i+1, joint_tf)
        joint_R = deepcopy(joint_tf[0:3, 0:3])
        # Get the position of this joint relative to the base joint 
        joint_rel_pos = base_pos - deepcopy(joint_tf[0:3, 3])
        all_joint_R_pos.append(np.hstack((joint_R.flatten(), joint_rel_pos)))
    
    state = np.hstack((orig_state, *all_joint_R_pos))
    return state

# Get the robot state with the link transforms for all 
def get_full_robot_state_modules(sim, robot_id, modules, rel_rot=False):
    orig_state = get_robot_state(sim, robot_id)
    n_dofs = sim.get_robot_dof_count(robot_id)

    # Iterate through each hinge joint and get their
    # corresponding link's rotation and position
    all_joint_R_pos_dct = {}
    for module in modules:
        module_base_pos = None
        module_base_rot = None
        for i in module:
            joint_tf = np.zeros((4, 4), order = 'f')
            sim.get_link_transform(robot_id, i+1, joint_tf)
            global_R = deepcopy(joint_tf[0:3, 0:3])
            global_pos = deepcopy(joint_tf[0:3, 3])
            
            if module_base_pos is None:
                module_base_pos = global_pos
            if module_base_rot is None:
                module_base_rot = global_R
            # Get the position of this joint relative to the module base position
            joint_rel_pos = module_base_pos - global_pos
            if rel_rot:
                joint_rel_R = np.matmul(module_base_rot.transpose(), global_R)
                all_joint_R_pos_dct[i] = np.hstack((joint_rel_R.flatten(), joint_rel_pos))
            else:
                all_joint_R_pos_dct[i] = np.hstack((global_R.flatten(), joint_rel_pos))
    all_joint_R_pos = []
    # Make sure that the order corresponds to the order of joints
    for i in range(n_dofs):
        all_joint_R_pos.append(all_joint_R_pos_dct[i])

    state = np.hstack((orig_state, *all_joint_R_pos))

    return state


def presimulate(robot):
    """Find an initial position that will place the robot on the ground behind the
    x=0 plane, and check if the robot collides in its initial configuration."""
    temp_sim = rd.BulletSimulation()
    temp_sim.add_robot(robot, np.zeros(3), rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    temp_sim.step()
    robot_idx = temp_sim.find_robot_index(robot)
    lower = np.zeros(3)
    upper = np.zeros(3)
    temp_sim.get_robot_world_aabb(robot_idx, lower, upper)
    return [-upper[0], -lower[1], 0.0], temp_sim.robot_has_collision(robot_idx)

def presimulate_with_offset(robot, offset_size, rng):
    """Find an initial position that will place the robot on the ground behind the
    x=0 plane, and check if the robot collides in its initial configuration."""
    pos = np.zeros(3)
    # Add a random offset to the starting x, z coordinates in order to 
    # generate different trajectories
    temp_sim = rd.BulletSimulation()
    # Generate random offset. Offset is in range 2 * offset_size * [-0.5, 0.5)
    pos[0], pos[1] = 2 * offset_size * (rng.random(2) - 0.5)
    temp_sim.add_robot(robot, pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    temp_sim.step()
    robot_idx = temp_sim.find_robot_index(robot)
    lower = np.zeros(3)
    upper = np.zeros(3)
    temp_sim.get_robot_world_aabb(robot_idx, lower, upper)
    return [-upper[0], -lower[1], 0.0], temp_sim.robot_has_collision(robot_idx)