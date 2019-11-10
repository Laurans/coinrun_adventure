import json

from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
from tf_explain.utils.display import heatmap_display
from coinrun import make
from coinrun_adventure.common import Metadata, Step
from coinrun_adventure.utils import mkdir
from loguru import logger

from tf_explain.utils.image import apply_grey_patch
import math


def _experimental(model_keras, obs, actions):

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


def experimental(model_keras, image, actions):
    class_index = int(actions)
    patch_size = 5
    image = image[0]

    sensitivity_map = np.zeros(
        (math.ceil(image.shape[0] / patch_size), math.ceil(image.shape[1] / patch_size))
    )
    patches = [
        apply_grey_patch(image, top_left_x, top_left_y, patch_size)
        for index_x, top_left_x in enumerate(range(0, image.shape[0], patch_size))
        for index_y, top_left_y in enumerate(range(0, image.shape[1], patch_size))
    ]

    coordinates = [
        (index_y, index_x)
        for index_x, _ in enumerate(range(0, image.shape[0], patch_size))
        for index_y, _ in enumerate(range(0, image.shape[1], patch_size))
    ]

    predictions = model_keras.raw_value(np.array(patches))[-1].numpy()
    target_class_predictions = [prediction[class_index] for prediction in predictions]

    for (index_y, index_x), confidence in zip(coordinates, target_class_predictions):
        sensitivity_map[index_y, index_x] = 1 - confidence

    heatmap = heatmap_display(sensitivity_map, image, cv2.COLORMAP_VIRIDIS)
    return heatmap


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
        sequence_folder="sequence",
        images_folder="image",
        explain_folder="explain",
    ).as_json()

    with open(str(destination / "metadata.json"), "w") as outfile:
        json.dump(metadata, outfile)

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
