import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import names

from coinrun_adventure.config import ExpConfig
from coinrun_adventure.ppo import get_model, learn
from coinrun_adventure.utils import common_arg_parser, mkdir, restore_model, setup
from coinrun_adventure.common import add_final_wrappers
from coinrun import make

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def main(args_list: list):
    arg_parser: ArgumentParser = common_arg_parser()
    args, _ = arg_parser.parse_known_args(args_list)

    if args.train:
        setup()

        env = make("standard", num_envs=ExpConfig.NUM_ENVS)
        env = add_final_wrappers(env)

        dirname = time.strftime("%Y%m%d_%H%M") + "_" + names.get_first_name()
        destination = ExpConfig.SAVE_DIR / dirname
        mkdir(destination)
        learn(destination, env)

    if args.test:
        experiment_folder = Path(args.exp).resolve()
        model = get_model()
        restore_model(model, experiment_folder)
        # TODO: Test the model on 3 environements

    if args.play and args.exp is not None:
        experiment_folder = Path(args.exp).resolve()
        model = get_model()
        restore_model(model, experiment_folder)
        play(experiment_folder, model)


if __name__ == "__main__":
    main(sys.argv)
