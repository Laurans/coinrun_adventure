import json
import sys
import time
from pathlib import Path

import names

from coinrun_adventure.common.data_structures import Metadata, Step
from coinrun_adventure.config import ExpConfig
from coinrun_adventure.utils.misc_util import mkdir
from coinrun_adventure.utils.setup_util import setup, common_arg_parser
from ppo.entrypoint import learn
from coinrun import make
import numpy as np


def play(destination, model):
    destination = Path(destination).resolve() / "play"
    sequence_folder = destination / "sequence"
    images_folder = destination / "image"
    mkdir(sequence_folder)
    mkdir(images_folder)

    metadata = Metadata(
        game_name="Coin run [OpenAI]",
        action_names=[
            "none",
            "right",
            "left",
            "jump",
            "right-jump",
            "left-jump",
            "down",
        ],
        sequence_folder="sequence",
        images_folder="image",
    ).as_json()

    with open(str(destination / "metadata.json"), "w") as outfile:
        json.dump(metadata, outfile)

    setup()
    env = make("standard", num_envs=1)

    obs = env.reset()
    timestep = 0
    episode_rew = 0
    done = False
    while not done:
        actions, _, _, _ = model.step(obs)

        next_obs, rew, done, _ = env.step(actions.numpy())
        done = done.any() if isinstance(done, np.ndarray) else done
        episode_rew += rew

        step = Step(
            timestep=timestep,
            imagename=f"{timestep:05d}.jpg",
            reward=rew,
            done=done,
            actions=actions,
        ).as_json()

        cv2.imwrite(f"{}")

        with open(str(images_folder / f"{timestep:05d}.json"), "w") as outfile:
            json.dump(step, outfile)

    env.close()


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

    if args.play and args.exp is not None:
        print("Play", args)
        play(args.exp)


if __name__ == "__main__":
    main(sys.argv)
