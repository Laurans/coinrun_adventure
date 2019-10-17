from coinrun_adventure.config import ExpConfig
from coinrun import setup_utils
import argparse


def setup(**kwargs):
    ExpConfig.merge(kwargs)
    setup_utils.setup(**kwargs)


def common_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--train", help="Train the agent", default=False, action="store_true"
    )
    parser.add_argument(
        "--play", help="Play an episode", default=False, action="store_true"
    )
    parser.add_argument("--exp", help="Folder path to the experience")

    return parser
