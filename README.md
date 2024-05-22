# MeMo: Learning Meaningful, Modular Controllers

This is the official codebase for MeMo: Learning Meaningful, Modular Controllers via Noise Injection

## Installing RoboGrammar / DiffHand 

First, cd to `rl_RoboGrammar/RoboGrammar` / `rl_Diffhand/DiffHand` and follow the RoboGrammar / DiffHand specific installation instructions.  While still in the RoboGrammar virtualenv / DiffHand conda environment, install the following prerequisites:

- Tensorflow: `pip install tensorflow`
- OpenAI Gym:  `pip install gym`

Then, cd to `rl_RoboGrammar` / `rl_DiffHand` and download the OpenAI Baselines:

  ```
  cd externals/baselines
  pip install -e .
  ```

Finally, cd to the `shared` folder and download pytorch-a2c-ppo-acktr-gail:

  ```
  cd externals/pytorch-a2c-ppo-acktr-gail
  pip install -e .
  ```

In addition, to try out Jacobian regularization from [1]:

  ```
  pip install git+https://github.com/facebookresearch/jacobian_regularizer
  ```

[1] Judy Hoffman, Daniel A. Roberts, and Sho Yaida, "Robust Learning with Jacobian Regularization," 2019. [arxiv:1908.02729](https://arxiv.org/abs/1908.02729)

## RL training

Scripts for RL and IL training are in the `design/` folders of `rl_RoboGrammar` and `rl_Diffhand`.

If you are in `rl_RoboGrammar` folder, below are the commands to train a standard MLP for 6 leg centipede, 6 leg worm, and 6 leg hybrid with RL:

### Centipede
```
python3 design/train.py --task FrozenLakeTask --num-processes 8 --seed 2   --robot-rank "6leg" --rule-seq "0, 7, 1, 12, 2, 3, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 3, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 6" --env-name RobotLocomotion-v2 --hi-mode None --base-hidden-size 128 --base-last-hidden-size 1024 --num-steps 2048 
```

### Worm
```
python3 design/train.py --task FrozenLakeTask --num-processes 8 --seed 2 --robot-rank "long_6leg" --rule-seq "0, 1, 1, 12, 12, 3, 3, 12, 12, 7, 2, 18, 8, 5, 1, 3, 12, 1, 3, 12, 1, 2, 12, 16, 4, 8, 18, 8, 5, 1, 3, 12, 1, 3, 12, 1, 2, 12, 16, 4, 8, 18, 8, 5, 6" --env-name RobotLocomotion-v2 --hi-mode None --base-hidden-size 128 --base-last-hidden-size 384 --num-steps 2048
```

### Hybrid
```
python3 design/train.py --task FrozenLakeTask --num-processes 8 --seed 2 --robot-rank "6leg-hybrid-v2" --rule-seq "0, 1, 1, 12, 12, 3, 3, 12, 12, 7, 2, 18, 8, 5, 1, 12, 3, 1, 12, 3, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 3, 1, 12, 3, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 6" --env-name RobotLocomotion-v2 --hi-mode None --base-hidden-size 128 --base-last-hidden-size 896 --num-steps 2048
```

### 4 Finger Claw

If you are in `rl_DiffHand` folder, to train a standard MLP architecture for the 4-finger claw grasping a cube :

```
python design/train.py --num-processes 16 --seed 2 --num-steps 100 --num-env-steps 300000 --env-name AllRelClawGraspCube-v0  --robot-rank "4_grasp_finger" --task "cube_arm_1" --base-hidden-size 64 --base-last-hidden-size 320 --hi-mode None
```

## IL training

In `rl_RoboGrammar` / `rl_DiffHand` folder, using a trained expert, collect the validation dataset. 

```
python3 design/collect.py --model-dir [MODEL_DIRECTORY] --n-train-episodes 500
```

`MODEL_DIRECTORY` is the save directory of a RL agent. Then, run DAgger with the validation dataset and the expert RL agent.

```
python3 design/train.py --task FrozenLakeTask --num-processes 8 --seed 2   --robot-rank "6leg" --rule-seq "0, 7, 1, 12, 2, 3, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 3, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 6" --env-name RobotLocomotion-v2 --hi-mode "v2" --base-hidden-size 128 --base-last-hidden-size 1024 --train-mode il --il-lr 2e-3 --il-batch-size 1024 --nonlinearity-mode "tanh" --n-train-episodes 500 --dagger-expert-dir [MODEL_DIRECTORY] --use-noise --noise-levels 1.0
```

## Structure Transfer 

We can transfer the modules from a pretrained modular architecture, saved at `PRETRAINED_MODEL_PATH`, as follows. Note that `--transfer-str` is typically used to organize experiment runs by algorithm and train morphology / task, e.g. `memo-6leg-frozen`.

### 6 to 12 Leg Centipede

```
python3 design/train.py --task FrozenLakeTask --num-processes 8 --seed 2 --robot-rank "12leg" --rule-seq "0, 7, 1, 12, 2, 3, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 3, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 3, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 3, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 3, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 6" --env-name RobotLocomotion-v2 --hi-mode v2 --base-hidden-size 128 --base-last-hidden-size 2176 --set-low-logstd -1.0 --transfer-str memo-6leg-frozen --num-steps 256 --load-model-path [PRETRAINED_MODEL_PATH]
```

### 6 to 10 Leg Worm

```
python3 design/train.py --task FrozenLakeTask --num-processes 8 --seed 2  --robot-rank "long_10leg" --rule-seq  "0, 1, 1, 12, 12, 3, 3, 12, 12, 7, 2, 18, 8, 5, 1, 3, 12, 1, 3, 12, 1, 2, 12, 16, 4, 8, 18, 8, 5, 1, 3, 12, 1, 3, 12, 1, 2, 12, 16, 4, 8, 18, 8, 5, 1, 3, 12, 1, 3, 12, 1, 2, 12, 16, 4, 8, 18, 8, 5, 1, 3, 12, 1, 3, 12, 1, 2, 12, 16, 4, 8, 18, 8, 5, 6" --env-name RobotLocomotion-v2 --set-low-logstd -1.0 --logstd-mode separate --num-steps 1024 --hi-mode v2 --base-hidden-size 128 --base-last-hidden-size 640 --transfer-str memo-long6-frozen --load-model-path [PRETRAINED_MODEL_PATH]
```

### 6 to 10 Leg Hybrid

```
python3 design/train.py --task FrozenLakeTask --num-processes 8 --seed 2 --robot-rank "10leg-hybrid-v2" --rule-seq "0, 1, 1, 12, 12, 3, 3, 12, 12, 7, 2, 18, 8, 5, 1, 12, 3, 1, 2, 18, 8, 5, 1, 12, 3, 1, 12, 3, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 3, 1, 12, 3, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 3, 1, 12, 3, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 6" --env-name RobotLocomotion-v2 --set-low-logstd -1.0 --num-steps 1024 --hi-mode v2 --base-hidden-size 128 --base-last-hidden-size 1408 --transfer-str memo-hybrid-frozen --load-model-path [PRETRAINED_MODEL_PATH]
```

### 4 to 5 finger Claw

```
python design/train.py --num-processes 16 --seed 2 --num-steps 100 --num-env-steps 300000 --env-name AllRelClawGraspCube-v0 --robot-rank "5_grasp_finger" --task "5_cube_arm_1" --hi-mode v2 --base-hidden-size 64 --base-last-hidden-size 384 --set-low-logstd -1.0 --transfer-str "memo-4finger-cube" --load-model-path [PRETRAINED_MODEL_PATH]
```

## Task Transfer

### Frozen to Gap

Below is the command to transfer the boss controller (via `--load-master`) and the modules from a 6 leg centipede traversing a Frozen Terrain to the Gap Terrain. To test transfer to different terrains, change the input to the `--task` argument, e.g. to `SteppedTerrainTask`. 

```
python3 design/train.py --task NewGapTerrainTask --num-processes 8 --seed 2   --robot-rank "6leg" --rule-seq "0, 7, 1, 12, 2, 3, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 1, 12, 3, 1, 12, 2, 9, 4, 16, 8, 14, 4, 15, 8, 4, 16, 8, 4, 18, 8, 5, 6" --env-name RobotLocomotion-v2 --hi-mode v2  --base-hidden-size 128 --base-last-hidden-size 1024 --transfer-str "memo-6leg-frozen" --set-low-logstd -1.0 --load-master --finetune-model --num-steps 2048 --load-model-path [PRETRAINED_MODEL_PATH]
```

### Cube to Sphere

```
python design/train.py --num-processes 16 --seed 2 --num-steps 100 --num-env-steps 300000 --env-name AllRelClawGraspSphere-v0  --robot-rank "4_grasp_finger" --task "sphere_arm_1" --hi-mode v2 --base-hidden-size 64 --base-last-hidden-size 320 --set-low-logstd -1.0 --transfer-str "memo-4finger-cube" --load-model-path  [PRETRAINED_MODEL_PATH]
```