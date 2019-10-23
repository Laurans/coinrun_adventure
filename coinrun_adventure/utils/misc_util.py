from pathlib import Path
import tensorflow as tf
import datetime
from typing import Union
from coinrun_adventure.utils.mpi_util import mpi_average_train_test
import numpy as np


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    return f"./log/{name}-{get_time_str()}"


def mkdir(path: Union[str, Path]):
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    if hasattr(obj, "close"):
        obj.close()


def save_model(model, save_path):
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, str(save_path), max_to_keep=None)
    manager.save()


def restore_model(model, save_path):
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(tf.train.latest_checkpoint(save_path))
    return model


def process_ep_buf(epinfobuf, sync_from_root):
    rewards = [epinfo["r"] for epinfo in epinfobuf]
    rew_mean = np.nanmean(rewards)

    if sync_from_root:
        rew_mean = mpi_average_train_test([rew_mean])[0]

    return rew_mean
