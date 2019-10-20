import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import names
import numpy as np
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
from tf_explain.utils.display import heatmap_display

from coinrun import make
from coinrun_adventure.common import Metadata, Step
from coinrun_adventure.common.mpi_util import setup_mpi_gpus
from coinrun_adventure.config import ExpConfig
from coinrun_adventure.ppo import get_model, learn
from coinrun_adventure.utils import common_arg_parser, mkdir, restore_model, setup
from loguru import logger

os.environ["TZ"] = "Europe/Paris"


def experimental(model_keras, obs, actions):

    model = model_keras.pi
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer("conv2d").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(obs)
        loss = predictions[:, int(actions)]

    grads = tape.gradient(loss, conv_outputs)

    guided_grad = (
        tf.cast(conv_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    )

    cams = GradCAM.generate_ponderated_output(conv_outputs, guided_grad)
    heatmaps = np.array(
        [
            heatmap_display(cam.numpy(), image, cv2.COLORMAP_VIRIDIS)
            for cam, image in zip(cams, obs)
        ]
    )
    return heatmaps[0]


def play(destination, model):
    destination = Path(destination).resolve() / "play"
    sequence_folder = destination / "sequence"
    images_folder = destination / "image"
    images_explain_folder = destination / "explain"
    mkdir(sequence_folder)
    mkdir(images_folder)
    mkdir(images_explain_folder)

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
        sequence_folder_name="sequence",
        images_folder_name="image",
        images_explain_name="explain",
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
        actions, state_value, pi_raw = model.get_all_values(obs)
        actions = actions.numpy()
        state_value = state_value.numpy()
        pi_raw = pi_raw.numpy()
        gram_cam_image = experimental(model.network, obs, actions)

        next_obs, rew, done, _ = env.step(actions)
        obs = next_obs

        done = done.any() if isinstance(done, np.ndarray) else done
        episode_rew += rew

        step = Step(
            timestep=timestep,
            imagename=f"{timestep:05d}.jpg",
            reward=float(rew),
            done=int(done),
            actions=list(map(int, actions)),
            state_value=float(state_value[0]),
            pi_raw=list(map(float, pi_raw[0])),
        )

        cv2.imwrite(
            f"{str(images_folder/step.imagename)}",
            cv2.cvtColor(obs_hires, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            f"{str(images_explain_folder/step.imagename)}",
            cv2.cvtColor(gram_cam_image, cv2.COLOR_RGB2BGR),
        )

        with open(str(sequence_folder / f"{timestep:05d}.json"), "w") as outfile:
            json.dump(step.as_json(), outfile)

        logger.info(f"Save step: {timestep}")
        timestep += 1

    env.close()


def main(args_list: list):
    arg_parser: ArgumentParser = common_arg_parser()
    args, _ = arg_parser.parse_known_args(args_list)

    if args.train:
        dirname = time.strftime("%Y%m%d_%H%M") + "_" + names.get_first_name()
        destination = ExpConfig.SAVE_DIR / dirname
        mkdir(destination)
        setup_mpi_gpus()
        learn(destination)

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
