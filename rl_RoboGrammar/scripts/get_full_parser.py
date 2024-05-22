import sys
import os
import pdb
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
from shared.scripts.arguments import get_parser

def get_full_parser():
    parser = get_parser()

    parser.add_argument(
        '--grammar-file',
        type=str,
        default="data/designs/grammar_apr30_labeled.dot",
        help='Grammar file (.dot)')
    parser.add_argument(
        '--norm-weight',
        type=float,
        default=0.7,
        help='Weight on the negative of the action norm in computing training reward')
    
    parser.add_argument(
        '--max-episode-length',
        type=int,
        default=128,
        help='Max episode length')
    
    parser.add_argument(
        '--rule-sequence',
        type=str,
        help='Design rule sequence to apply')

    return parser