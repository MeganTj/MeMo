import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='RL')
    
    # robot_locomotion parameters
    parser.add_argument(
        '--task',
        type=str,
        default='FlatTerrainTask',
        help='different tasks/terrains: FlatTerrainTask | RidgedTerrainTask | GapTerrainTask | StairsTerrainTask | FrozenLakeTask | HillTerrainTask'
        'for grasping: cube | 5_cube | sphere')
    parser.add_argument(
        '--robot-rank',
        type=str,
        help='A unique identifier for the robot. Currently just used in rendering')
    parser.add_argument(
        '--render-only',
        action='store_true',
        default=False,
        help='Only do rendering')
    parser.add_argument(
        '--train-mode',
        type = str,
        choices=["rl", "il", "ni-err", "sum-obj"],
        help='Whether to use RL or IL, or evaluate sum of two objectives',
        default="rl")
    # logging
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
        
    # default RL args
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--il-dataset-dir',
        default='',
        help='directory that contains expert demonstrations for il')
    parser.add_argument(
        '--il-subsample-frequency',
        type=int,
        default=1,
        help='subsample every k frames')
    parser.add_argument(
        '--il-num-trajectories',
        type=int,
        default=4,
        help='Number of trajectories to use for the expert dataset')
    parser.add_argument(
        '--il-batch-size',
        type=int,
        default=128,
        help='il batch size (default: 128)')
    parser.add_argument(
        '--il-epoch', type=int, default=1, help='il epochs (default: 1)')
    parser.add_argument(
        '--il-model-idx', type=int, default=-1, help="Which model to use for IL. Default is the latest")
    parser.add_argument(
        '--il-lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--il-est-error', action="store_true", default=False, help='Whether or not to estimate error')
    parser.add_argument(
        '--il-lr-epochs', type=str, help='epochs at which to decay learning rate')
    parser.add_argument(
        '--il-lr-gamma', type=float, default=0.5, help='decay factor')
    parser.add_argument(
        '--lr', type=float, default=3e-4, help='learning rate (default: 3e-4)')
    # Arguments for dagger
    parser.add_argument(
            '--dagger-epoch',
            type = int, default = 175,
            help='# of epochs for dagger')
    parser.add_argument(
            '--n-dagger-episodes',
            type = int, default = 1,
            help='# of episodes to sample from model on each iteration of dagger')
    # Arguments for training post dagger
    parser.add_argument(
            '--post-dag-noise',
            action='store_true',
            default=False, 
            help='Whether or not to train with noise after dagger')
    parser.add_argument(
            '--post-dag-epoch',
            type = int, default = 1,
            help='# of epochs to train after all dagger data points are added')
    # Arguments for dagger
    parser.add_argument(
            '--dagger-expert-dir',
            type = str, default = None,
            help='Directory of the expert model')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--master-norm-weight',
        type=float,
        default=0.,
        help='Weight on the negative of the norm of the master output in computing training reward')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--dataset-seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument(
        '--dataset-offset-size', type=float, default=0.0, help='random seed (default: 1)')
    parser.add_argument(
        '--dagger-seed', type=int, default=20000, help='random seed for testing (default: 1)')
    parser.add_argument(
        '--test-seed', type=int, default=10000, help='random seed for testing (default: 1)')
    parser.add_argument(
        '--test-offset-size', type=float, default=5.0, help='how much to offset the initial position when evaluating')
    parser.add_argument(
        '--n-train-episodes', type=int, default=100, help='how many training episodes for imitation learning')
    parser.add_argument(
        '--n-eval-episodes', type=int, default=200, help='how many episodes for evaluation')
    parser.add_argument(
        '--hierarchy-dir', type=str, default="data/hierarchies", help='how much to offset the initial position when evaluating')
    parser.add_argument(
        '--hierarchy-suffix', type=str, default="", help='suffix to add to the end of the hierarchy json filename')
    parser.add_argument(
        "--hi-mode",
        type=str,
        default="None",
        choices=["None", "v2"],
        help="Which hierarchical architecture to use",
    )
    parser.add_argument(
        '--extend-local-state',
        action='store_true',
        default=False)
    parser.add_argument(
        '--share-within-mod',
        action='store_true',
        default=False, help='If True, share one signal and decoder network for all joints in a module')
    parser.add_argument(
        '--decoder-hidden-size', type=int, default=-1, help='Sizes of embeddings passed between hierarchies')
    parser.add_argument(
        '--module-hidden-size', type=int, default=32, help='Size of module hidden layer')
    parser.add_argument(
        '--decoder-num-hidden-layers', type=int, default=2, help='Number of hidden layers in hierarchical decoder')
    parser.add_argument(
        '--nonlinearity-mode', default="tanh",
        choices=["tanh", "relu", "elu"],
        help='type of nonlinearity to use')
    # Arguments for the base layer
    parser.add_argument(
        '--base-hidden-size', type=int, default=64, help='Number of hidden units to use in base network')
    parser.add_argument(
        '--base-last-hidden-size', type=int, default=-1, help='Number of hidden units to use in last layer of base network')
    parser.add_argument(
        '--base-num-hidden-layers', type=int, default=2, help='Number of hidden layers in base network')
    # Whether or not to share logstd
    parser.add_argument(
        '--logstd-mode',
        type=str,
        default="separate",
        choices=["shared", "separate"],
        help="l1 regularization weight on master output")
    # Argument for the hierarchical linear decoder
    parser.add_argument(
            '--sqrt-gain',
            action='store_true',
            default=False)
    # L1 reg in IL
    parser.add_argument(
        '--l1-weight',
        type=float,
        default=0.,
        help="l1 regularization weight on master output")
    # L2 reg in IL
    parser.add_argument(
        '--l2-weight',
        type=float,
        default=0.,
        help="l2 regularization weight on master output")
    # Jacobian reg in IL
    parser.add_argument(
        '--jac-weight',
        type=float,
        default=0.,
        help="jacobian regularization weight on modules")
    # Jacobian reg in IL
    parser.add_argument(
        '--jac-nproj',
        type=int,
        default=1,
        help="jacobian regularization # of projections")
    # Noise arguments 
    parser.add_argument(
            '--use-noise',
            action='store_true',
            default=False)
    parser.add_argument(
            '--noise-levels',
            default="0",
            type=str,
            help="levels of noise to add to activations")
    parser.add_argument(
        '--noise-relative',
        action='store_true',
        default=False,
        help="Whether the noise level is scaled by the input value")
    parser.add_argument(
        '--noise-train',
        action='store_true',
        default=False,
        help="Whether the noise level is trainable")
    parser.add_argument(
        '--sni-mode',
        type=str,
        default="None",
        choices=["None", "rollout", "all"],
        help="Whether or not to use SNI when injecting noise during RL")
    parser.add_argument(
        '--linear-noise-decay',
        action='store_true',
        default=False,
        help="Whether to decay the noise level linearly")
    parser.add_argument(
        '--end-noise',
        type=float,
        default=0.25,
        help="If decaying the noise, specifies the end noise level")
    parser.add_argument(
            '--noisy-min-clip',
            type=float,
            default=None,
            help="Minimum value to clip the noisy input to")
    parser.add_argument(
            '--noisy-max-clip',
            type=float,
            default=None,
            help="Maximum value to clip the noisy input to")
    parser.add_argument(
        "--report-frequency",
        type=int,
        default=-1,
        help="Report every K batches. If -1, just report at the end of an epoch",
    )
    parser.add_argument(
        "--loss-mode",
        type=str,
        default="log_prob",
        choices=["log_prob", "mse", "noise-sum", "noise-lp", "noise-mse"],
        help="What loss to use when training IL",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="loss",
        help="What metric to use to update sparsity.",
    )
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--set-low-logstd',
        type=float,
        default=None,
        help='If not None, set the logstd of loaded policies to the given value')
    parser.add_argument(
        '--transfer-str',
        type=str,
        default=None, 
        help='string to differentiate transfer task experiments')
    parser.add_argument(
        '--transfer-modules',
        type=str,
        default=None, 
        help='string stating which modules to transfer, where the modules are comma separated, e.g. "0,1,2"')
    parser.add_argument(
        '--load-model',
        type=bool,
        default=False)
    parser.add_argument(
        '--load-model-path',
        type=str,
        default=None)
    parser.add_argument(
        '--finetune-model',
        action='store_true',
        default=False)
    parser.add_argument(
        '--load-master',
        action='store_true',
        default=False)
    parser.add_argument(
        '--load-ob-rms',
        action='store_true',
        default=False)
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--eval-num',
        type=int,
        default=1,
        help='number of fitness evaluation times')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=int(8e6),
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='RobotLocomotion-v0',
        help='environment to train on (default: RobotLocomotion-v0)')
    parser.add_argument(
        '--hier-env-name',
        default="None",
        help='Hierarchical environment to train on')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-pref', type = str, default = './trained_models')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--append-time-stamp',
        action='store_true',
        default=False,
        help='whether or not to append time stamp to the save directory')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--lr-schedule',
        type=str,
        default="decay",
        choices=["decay", "constant"],
        help='type of schedule on the learning rate')
    parser.add_argument(
        '--layernorm', 
        action='store_true',
        default=False,
        help='if use layernorm')
    parser.add_argument(
        '--render-interval',
        type=int,
        default=-1)
    parser.add_argument(
        '--device',
        type=str,
        default="cpu")

    return parser