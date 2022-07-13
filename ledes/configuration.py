import argparse
import os
from pecos.xmc import PostProcessor
import yaml
from pathlib import Path
import sys
from pecos.utils import smat_util, logging_util
from pecos.utils import cli
from pecos.core import XLINEAR_SOLVERS
import logging


def parse_evaluation_arguments():
    """Parse evaluation arguments"""

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "-x",
        "--inst-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the npz file of the feature matrix (CSR, nr_insts * nr_feats)",
    )

    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        required=True,
        metavar="DIR",
        help="path to the model folder.",
    )
    parser.add_argument(
        "-om",
        "--overlap-model-folder",
        type=str,
        required=False,
        metavar="DIR",
        help="path to the overlap model folder(after running the new algorithm).",
    )

    parser.add_argument(
        "-k",
        "--topk",
        "--only-topk",
        type=int,
        default=10,
        dest="topk",
        metavar="INT",
        help="evaluate @k",
    )
    parser.add_argument(
        "-b",
        "--beam-size",
        type=int,
        default=None,
        metavar="INT",
        help="override the beam size specified in the model (default None to disable overriding)",
    )

    parser.add_argument(
        "-pp",
        "--post-processor",
        type=str,
        choices=PostProcessor.valid_list(),
        default=None,
        metavar="STR",
        help="override the post processor specified in the model (default None to disable overriding)",
    )

    parser.add_argument(
        "-y",
        "--label-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the npz file of the label matrix (CSR, nr_insts * nr_labels)",
    )

    parser.add_argument(
        "-o",
        "--save-pred-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to save the predictions (sorted CSR, nr_insts * nr_labels)",
    )

    parser.add_argument(
        "-B",
        "--max-pred-chunk",
        default=10**7,
        metavar="INT",
        type=int,
        help="Max number of instances to predict on at once, set to avoid OOM. Set to None to predict on all instances at once. Default 10^7",
    )

    parser.add_argument(
        "-n",
        "--threads",
        type=int,
        default=-1,
        metavar="THREADS",
        help="number of threads to use (default -1 to denote all the CPUs)",
    )

    parser.add_argument(
        "-so",
        "--selected-output",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the npz file of the selected output matrix (CSR, nr_insts * nr_labels), only-topk and beam-size are ignored if given",
    )
    parser.add_argument(
        "--mapper",
        type=str,
        default=None,
        help="path to pseudo label mapper. If None, this variable is ignored.",
    )
    parser.add_argument(
        "--unused-labels",
        type=str,
        default=None,
        help="path to unused label set. If None, this variable is ignored.",
    )

    args, unknown = parser.parse_known_args()
    return args

def parse_reclusterer_arguments():
    parser = argparse.ArgumentParser(
        prog="Reorganize the clusters, move some of the labels around to "
        "improve recall."
    )
    parser.add_argument(
        "-x",
        "--inst-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to npz file of feature matrix",
    )
    parser.add_argument(
        "-y",
        "--label-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the npz file of the label matrix",
    )
    parser.add_argument(
        "-m",
        "--model-folder",
        type=lambda p: os.path.abspath(p),
        required=True,
        metavar="DIR",
        help="path to the model folder",
    )
    parser.add_argument(
        "-o",
        "--model-folder-output",
        type=lambda p: os.path.abspath(p),
        required=True,
        metavar="DIR",
        help="path to the model output folder",
    )
    parser.add_argument(
        "-b",
        "--beam-size",
        type=int,
        required=True,
        help="Beam size to calculate the matching matrix.",
    )
    parser.add_argument(
        "--n_copies", type=int, default=2, help="number of copies for each label(lambda).",
    )
    args, unknown = parser.parse_known_args()
    return args

def parse_train_arguments():
    """Parse training arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generate-params-skeleton",
        action="store_true",
        help="generate template params-json to stdout",
    )

    skip_training = "--generate-params-skeleton" in sys.argv
    # ========= parameter jsons ============
    parser.add_argument(
        "--params-path",
        type=str,
        default=None,
        metavar="PARAMS_PATH",
        help="Json file for params (default None)",
    )
    # ======= actual arguments ========

    # Required parameters
    parser.add_argument(
        "-x",
        "--inst-path",
        type=str,
        required=not skip_training,
        metavar="PATH",
        help="path to the CSR npz or Row-majored npy file of the feature matrix (nr_insts * nr_feats)",
    )

    parser.add_argument(
        "-y",
        "--label-path",
        type=str,
        required=not skip_training,
        metavar="PATH",
        help="path to the CSR npz file of the label matrix (nr_insts * nr_labels)",
    )

    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        required=not skip_training,
        metavar="DIR",
        help="path to the model folder.",
    )

    # Optional

    # Indexing parameters
    parser.add_argument(
        "-f",
        "--label-feat-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the CSR npz or Row-majored npy file of the label feature matrix (nr_labels * nr_label_feats)",
    )

    parser.add_argument(
        "--nr-splits",
        type=int,
        default=16,
        metavar="INT",
        help="number of splits used to construct hierarchy (a power of 2 is recommended)",
    )

    parser.add_argument(
        "--max-leaf-size",
        type=int,
        default=100,
        metavar="INT",
        help="The max size of the leaf nodes of hierarchical 2-means clustering. If larger than total number of labels, One-Versus-All model will be trained. Default 100.",
    )

    parser.add_argument(
        "--imbalanced-ratio",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Value between 0.0 and 0.5 (inclusive). Indicates how relaxed the balancedness constraint of 2-means can be. Specifically, if an iteration of 2-means is clustering L labels, the size of the output 2 clusters will be within approx imbalanced_ratio * 2 * L of each other. (default 0.0)",
    )

    parser.add_argument(
        "--imbalanced-depth",
        type=int,
        default=100,
        metavar="INT",
        help="After hierarchical 2-means clustering has reached this depth, it will continue clustering as if --imbalanced-ratio is set to 0.0. (default 100)",
    )

    parser.add_argument(
        "--spherical",
        type=cli.str2bool,
        metavar="[true/false]",
        default=True,
        help="If true, do l2-normalize cluster centers while clustering. Default true.",
    )

    parser.add_argument(
        "--seed", type=int, default=0, metavar="INT", help="random seed (default 0)"
    )

    parser.add_argument(
        "--kmeans-max-iter",
        type=int,
        default=20,
        metavar="INT",
        help="max number of k-means iterations for indexer (default 20)",
    )

    parser.add_argument(
        "-n",
        "--threads",
        type=int,
        default=-1,
        metavar="INT",
        help="number of threads to use (default -1 to denote all the CPUs)",
    )

    parser.add_argument(
        "-c",
        "--code-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the npz file of the code matrix (CSC, nr_labels * nr_codes)",
    )

    parser.add_argument(
        "-r",
        "--rel-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the npz file of the relevance matrix (CSR, nr_insts * nr_labels). Should have same sparsity pattern as label matrix.",
    )

    parser.add_argument(
        "--rel-norm",
        type=str,
        choices=["l1", "l2", "max", "no-norm"],
        default="no-norm",
        metavar="STR",
        help="norm type to row-wise normalzie relevance matrix for cost-sensitive learning",
    )

    parser.add_argument(
        "--rel-mode",
        type=str,
        metavar="STR",
        default="disable",
        help="mode to use relevance score for cost sensitive learning ['disable'(default), 'induce', 'ranker-only']",
    )

    parser.add_argument(
        "-um",
        "--usn-match-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the npz file of the user supplied matching matrix (CSR, nr_insts * nr_codes), will be add to negative sampling if given",
    )

    parser.add_argument(
        "-uy",
        "--usn-label-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the npz file of the user supplied label importance matrix (CSR, nr_insts * nr_labels), will be add to negative sampling if given",
    )

    # Linear matching/ranking parameters
    parser.add_argument(
        "-s",
        "--solver-type",
        type=str,
        default="L2R_L2LOSS_SVC_DUAL",
        metavar="STR",
        help="{} (default L2R_L2LOSS_SVC_DUAL)".format(" | ".join(XLINEAR_SOLVERS.keys())),
    )

    parser.add_argument(
        "--Cp",
        type=float,
        default=1.0,
        metavar="VAL",
        help="coefficient for positive class in the loss function (default 1.0)",
    )

    parser.add_argument(
        "--Cn",
        type=float,
        default=1.0,
        metavar="VAL",
        help="coefficient for negative class in the loss function (default 1.0)",
    )

    parser.add_argument(
        "--bias", type=float, default=1.0, metavar="VAL", help="bias term (default 1.0)"
    )

    parser.add_argument(
        "-ns",
        "--negative-sampling",
        type=str,
        choices=["tfn", "man", "tfn+man", "usn", "usn+tfn", "usn+man", "usn+tfn+man"],
        default="tfn",
        metavar="STR",
        dest="neg_mining_chain",
        help="Negative Sampling Schemes",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.1,
        metavar="VAL",
        help="threshold to sparsify the model weights (default 0.1)",
    )

    parser.add_argument(
        "-z",
        "--max-nonzeros-per-label",
        type=int,
        default=0,
        metavar="NONZEROS",
        help="keep at most NONZEROS weight parameters per label in model(default 0 to denote nr_features + 1)",
    )

    # Prediction kwargs
    parser.add_argument(
        "-k",
        "--only-topk",
        type=int,
        default=None,
        metavar="INT",
        help="the default number of top labels used in the prediction",
    )

    parser.add_argument(
        "-b",
        "--beam-size",
        type=int,
        default=10,
        metavar="INT",
        help="the default size of beam search used in the prediction",
    )

    parser.add_argument(
        "-pp",
        "--post-processor",
        type=str,
        choices=PostProcessor.valid_list(),
        default=None,
        metavar="STR",
        help="the default post processor used in the prediction",
    )

    parser.add_argument(
        "--verbose-level",
        type=int,
        choices=logging_util.log_levels.keys(),
        default=1,
        metavar="INT",
        help=f"the verbose level, {', '.join([str(k) + ' for ' + logging.getLevelName(v) for k, v in logging_util.log_levels.items()])}. Default 1",
    )

    args, unknown = parser.parse_known_args()
    return args

def get_configs(path = None):
    if path is None:
        path = 'configure/datasets/amazon-670k.yaml'
    data = dict()
    with open(path) as f:
        data = yaml.load(f, Loader = yaml.loader.SafeLoader)
    return data

def generate_configs_files(args):
    pass

if __name__ == '__main__':
    ev_args = parse_evaluation_arguments()
    train_args = parse_train_arguments()
    print(train_args)

