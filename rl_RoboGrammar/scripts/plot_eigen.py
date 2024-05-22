"""
Script for plotting normalized eigenvalue distributions
"""
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
import numpy as np
import argparse
from scipy.stats import skew

import pdb


def plot_eigenvalues(all_eigen, all_titles, save_path, svd_key = "master_svd", 
                     buffer=100):
    files = all_eigen.split(",")
    titles = all_titles.split(";")
    
    fig, ax = plt.subplots(1, len(files), squeeze=False, figsize = (10 * len(files), 8))
    fig.text(0.5, 0.025, 'Normalized Eigenvalue', ha='center', va='center', fontsize=42)
    fig.text(0.01, 0.5, 'Count', ha='center', va='center', rotation='vertical', fontsize=42)
    ylim = None
    max_y = None
    for idx, (file, title) in enumerate(zip(files, titles)):
        npzfile = np.load(file, allow_pickle=True)
        eigenvalues = npzfile[svd_key]
        print(f'Scipy Skew for {title}: {skew(eigenvalues)}')
        ax[0][idx].hist(eigenvalues, args.n_bins, range=(0, 1.0))
        ax[0][idx].tick_params(labelsize=34)
        ax[0][idx].set_title(title, fontsize=42)
        ax[0][idx].set_xlim([0, 1])
        if ylim is None or ylim[1] + buffer > max_y:
            ylim=ax[0][idx].get_ylim()
            ylim = (ylim[0], ylim[1] + buffer)
            max_y = ylim[1] + buffer
        if idx > 0:
            ax[0][idx].set_yticklabels([])
    
    for idx in range(len(files)):
        ax[0][idx].set_ylim(ylim)
    print(save_path)
    plt.tight_layout(rect=[0.02, 0.05, 1, 1])
    plt.savefig(save_path)

def plot_single(files, save_path, svd_key = "master_svd"):
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize = (10, 8))
    colors = ["purple", "blue"]
    labels = ["Memo", "MeMo (No NI)"]
    ax[0][0].set_xlim([0, 1])
    ax[0][0].set_xlabel('Normalized Eigenvalue', fontsize=32)
    ax[0][0].set_ylabel('Count', fontsize=32)
    for idx, file in enumerate(files):
        npzfile = np.load(file, allow_pickle=True)
        eigenvalues = npzfile[svd_key]
        ax[0][0].hist(eigenvalues, args.n_bins, range=(0, 1.0), histtype="step", 
                color=colors[idx], linewidth=3.0)
        ax[0][0].tick_params(labelsize=28)
    print(save_path)
    handle1 = matplotlib.lines.Line2D([], [], c=colors[0])
    handle2 = matplotlib.lines.Line2D([], [], c=colors[1])
    fig.legend([handle1, handle2], labels, bbox_to_anchor=(0.22, 1.), loc='upper left', fancybox=True, framealpha=0.5, fontsize=26)
    plt.tight_layout(rect=[0.0, 0.0, 1, 1])
    plt.savefig(save_path)

parser = argparse.ArgumentParser()
parser.add_argument('--save-pref', type = str, required=True)
parser.add_argument('--files', type = str, default="Ours,Standard")
parser.add_argument('--files-idx', type = str, default=None, help="If this is specifed, plot two experiments on a single plot")
parser.add_argument('--titles', type = str, default="Ours,Standard")
parser.add_argument('--n-bins', type = int, default=20)


args = parser.parse_args()
plot_eigenvalues(args.files, args.titles, os.path.join(f"{args.save_pref}_master_actuator_eigen.png"), 
                 svd_key = "master_svd")
if args.files_idx is not None:
    files = [args.files.split(",")[int(idx)] for idx in args.files_idx.split(",")]
    plot_single(files, os.path.join(f"{args.save_pref}_master_actuator_eigen_single.png"), 
                    svd_key = "master_svd")

