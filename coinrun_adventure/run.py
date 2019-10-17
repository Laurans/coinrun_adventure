import json
import sys
import time
from pathlib import Path

import names

from coinrun_adventure.common.data_structures import Metadata, Step
from coinrun_adventure.config import ExpConfig
from coinrun_adventure.utils.misc_util import mkdir, restore_model
from coinrun_adventure.utils.setup_util import setup, common_arg_parser
from ppo.entrypoint import learn, get_model
from coinrun import make
import numpy as np
import cv2


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

    setup(is_high_res=True)
    env = make("standard", num_envs=1)

    obs = env.reset()
    timestep = 0
    episode_rew = 0
    done = False
    while not done:
        obs_hires = env.render(mode="rgb_array")
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
        )

        cv2.imwrite(
            f"{str(images_folder/step.imagename)}",
            cv2.cvtColor(obs_hires, cv2.COLOR_RGB2BGR),
        )

        with open(str(sequence_folder / f"{timestep:05d}.json"), "w") as outfile:
            json.dump(step.as_json(), outfile)

    env.close()


def main(args):
    arg_parser = common_arg_parser()
    args, _ = arg_parser.parse_known_args(args)

    if args.train:
        dirname = names.get_first_name() + "_" + time.strftime("%Y%m%d%H%M")
        destination = ExpConfig.SAVE_DIR / dirname
        mkdir(destination)
        learn(destination)

    if args.test:
        experiment_folder = Path(args.exp).resolve()
        model = get_model()
        restore_model(model, experiment_folder)

    if args.play and args.exp is not None:
        experiment_folder = Path(args.exp).resolve()
        model = get_model()
        restore_model(model, experiment_folder)
        play(experiment_folder, model)


if __name__ == "__main__":
    main(sys.argv)
