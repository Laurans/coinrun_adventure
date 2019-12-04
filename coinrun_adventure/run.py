import os
import random
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import names
import torch.distributed as dist
import torch.multiprocessing as mp
from coinrun import make
from coinrun_adventure.common import add_final_wrappers
from coinrun_adventure.config import ExpConfig
from coinrun_adventure.play_with_agent import play
from coinrun_adventure.ppo.entrypoint import get_model, learn
from coinrun_adventure.test_with_agent import test
from coinrun_adventure.utils import (
    cleanup,
    common_arg_parser,
    mkdir,
    restore_model,
    setup,
)


def multi_setup(rank, world_size, destination):

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    setup()
    env = make("standard", num_envs=ExpConfig.NUM_ENVS)
    env = add_final_wrappers(env)

    learn(rank, destination, env)

    cleanup()


def train(args):

    dirname = time.strftime("%Y%m%d_%H%M") + "_" + names.get_first_name()
    destination = ExpConfig.SAVE_DIR / dirname
    mkdir(destination)

    rand_port = random.SystemRandom().randint(1000, 2000)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(rand_port)

    mp.spawn(
        multi_setup,
        args=(ExpConfig.WORLD_SIZE, destination),
        nprocs=ExpConfig.WORLD_SIZE,
        join=True,
    )


def main(args_list: list):
    arg_parser: ArgumentParser = common_arg_parser()
    args, _ = arg_parser.parse_known_args(args_list)

    if args.train:
        train(args)

    if args.test and args.exp is not None:
        experiment_folder = Path(args.exp).resolve()
        model = get_model()
        restore_model(model, experiment_folder)
        test(model)

    if args.play and args.exp is not None:
        experiment_folder = Path(args.exp).resolve()
        setup(
            is_high_res=True,
            num_levels=0,
            use_data_augmentation=False,
            high_difficulty=True,
            num_envs=1,
            rand_seed=17227,
            architecture="impala",
        )
        model = get_model()
        restore_model(model, experiment_folder)
        play(experiment_folder, model)


if __name__ == "__main__":
    main(sys.argv)
