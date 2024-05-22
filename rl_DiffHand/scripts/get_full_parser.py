import sys
import os
import pdb

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
from shared.scripts.arguments import get_parser

def get_full_parser():
    parser = get_parser()

    parser.add_argument(
        '--dist-weight',
        type=float,
        default=0.1,
        help='Weight on the negative of the fingertip distances to object')
    parser.add_argument(
        '--pos-dist-weight',
        type=float,
        default=3.5,
        help='Weight on the minimum fingertip distance to object. Relevant for placing in bin task')
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=5,
        help='Environment frame skip')
    parser.add_argument(
        '--norm-weight',
        type=float,
        default=0.,
        help='Weight on the negative of the action norm in computing training reward')
    parser.add_argument(
        '--max-episode-length',
        type=int,
        default=100,
        help='Max episode length')

    return parser