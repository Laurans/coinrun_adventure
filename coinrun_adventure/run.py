import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import names

from coinrun_adventure.config import ExpConfig
from coinrun_adventure.ppo import get_model, learn
from coinrun_adventure.utils import common_arg_parser, mkdir, restore_model, setup
from coinrun_adventure.common import add_final_wrappers
from coinrun_adventure.test_agent import play

from coinrun import make

from baselines.common import set_global_seeds
from baselines.common.mpi_util import setup_mpi_gpus
from mpi4py import MPI

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def train(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)
    setup_mpi_gpus()

    setup()

    env = make("standard", num_envs=ExpConfig.NUM_ENVS)
    env = add_final_wrappers(env)
    dirname = time.strftime("%Y%m%d_%H%M") + "_" + names.get_first_name()
    destination = ExpConfig.SAVE_DIR / dirname
    mkdir(destination)

    learn(destination, env)


def main(args_list: list):
    arg_parser: ArgumentParser = common_arg_parser()
    args, _ = arg_parser.parse_known_args(args_list)

    if args.train:
        train(args)

    if args.test:
        experiment_folder = Path(args.exp).resolve()
        model = get_model()
        restore_model(model, experiment_folder)
        # TODO: Test the model on 3 environements

    if args.play and args.exp is not None:
        experiment_folder = Path(args.exp).resolve()
        setup(
            is_high_res=True,
            num_levels=0,
            use_data_augmentation=0,
            high_difficulty=0,
            num_envs=1,
            set_seed=65,
            sync_from_root=False,
        )
        model = get_model()
        restore_model(model, experiment_folder)
        play(experiment_folder, model)


if __name__ == "__main__":
    main(sys.argv)
