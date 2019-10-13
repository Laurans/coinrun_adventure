import sys
from coinrun_adventure.utils.setup_util import common_arg_parser

from coinrun_adventure.utils.misc_util import mkdir
import time

from ppo.entrypoint import learn
import names
from coinrun_adventure.config import ExpConfig


def main(args):
    arg_parser = common_arg_parser()
    args, _ = arg_parser.parse_known_args(args)

    if args.train:
        print("Train", args)
        dirname = names.get_first_name() + "_" + time.strftime("%Y%m%d%H%M")
        destination = ExpConfig.SAVE_DIR / dirname
        print(destination)
        mkdir(destination)
        learn(destination)

    if args.play:
        print("Play", args)


if __name__ == "__main__":
    main(sys.argv)
