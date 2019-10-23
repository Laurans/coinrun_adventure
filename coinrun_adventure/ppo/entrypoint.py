from coinrun_adventure.utils import setup_util, misc_util
from coinrun_adventure.config import ExpConfig
from coinrun_adventure.ppo.model import Model
from coinrun_adventure.ppo.agent import PPORunner
import time
import numpy as np
import tensorflow as tf
from coinrun_adventure.logger import get_metric_logger, Logger
from pathlib import Path
from coinrun.common.vec_env import VecEnv
from loguru import logger as logo
from collections import deque
from mpi4py import MPI


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def run_update(update: int, nupdates: int, runner: PPORunner, model: Model):
    frac = 1.0 - (update - 1.0) / nupdates

    # Calculate the learning rate
    lrnow = ExpConfig.LR_FN(frac)
    cliprangenow = ExpConfig.CLIP_RANGE_FN(frac)

    # Get minibatch
    obs, returns, masks, actions, values, neglogpacs, epinfos = runner.run()

    # For each minibatch we'll calculate the loss and append it
    mblossvals = []
    # Index of each element of batchsize
    inds = np.arange(ExpConfig.NBATCH)
    for _ in range(ExpConfig.NUM_OPT_EPOCHS):
        # Randomize the indexes
        np.random.shuffle(inds)
        # 0 to batch size with batch train size step
        for start in range(0, ExpConfig.NBATCH, ExpConfig.NBATCH_TRAIN):
            end = start + ExpConfig.NBATCH_TRAIN
            mbinds = inds[start:end]
            slices = (
                tf.constant(arr[mbinds])
                for arr in (obs, returns, masks, actions, values, neglogpacs)
            )
            mblossvals.append(model.train(lrnow, cliprangenow, *slices))

    # Feedforward --> get losses --> updates
    lossvals = np.mean(mblossvals, axis=0)
    return lossvals, epinfos


def get_model():
    # x_input = tf.keras.Input(shape=input_shape, dtype=tf.uint8)

    model_fn = Model

    model = model_fn(
        ob_shape=ExpConfig.OB_SHAPE,
        ac_space=ExpConfig.AC_SPACE,
        policy_network_archi=ExpConfig.ARCHITECTURE,
        ent_coef=ExpConfig.ENTROPY_WEIGHT,
        vf_coef=ExpConfig.VALUE_WEIGHT,
        max_grad_norm=ExpConfig.MAX_GRAD_NORM,
        mode_sync_from_root=ExpConfig.SYNC_FROM_ROOT,
    )

    return model


def learn(exp_folder_path: Path, env: VecEnv):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    metric_logger: Logger = get_metric_logger(folder=exp_folder_path)

    model: Model = get_model()

    runner: PPORunner = PPORunner(
        env=env,
        model=model,
        num_steps=ExpConfig.NUM_STEPS,
        gamma_coef=ExpConfig.GAMMA,
        lambda_coef=ExpConfig.LAMBDA,
    )

    epinfobuf10 = deque(maxlen=10)
    epinfobuf100 = deque(maxlen=100)
    active_ep_buf = epinfobuf100
    mean_rewards = []
    datapoints = []
    can_save = True

    if ExpConfig.SYNC_FROM_ROOT and rank != 0:
        can_save = False

    tfirststart = time.perf_counter()

    nupdates = int(ExpConfig.TOTAL_TIMESTEPS // ExpConfig.NBATCH)

    logo.info(f"Start experiment, rank {rank}, can save {can_save}")
    for update in range(1, nupdates + 1):
        assert ExpConfig.NBATCH % ExpConfig.NUM_MINI_BATCH == 0
        # Start timer
        tstart = time.perf_counter()

        # Run an update
        lossvals, epinfos = run_update(update, nupdates, runner, model)
        epinfobuf10.extend(epinfos)
        epinfobuf100.extend(epinfos)

        # End timer
        tnow = time.perf_counter()

        # Calculate the fps
        fps = int(ExpConfig.NBATCH / (tnow - tstart))

        if update % ExpConfig.LOG_INTERVAL == 0 or update == 1:
            step = update * ExpConfig.NBATCH
            rew_mean_10 = misc_util.process_ep_buf(
                active_ep_buf, ExpConfig.SYNC_FROM_ROOT
            )
            ep_len_mean = np.nanmean([epinfo["l"] for epinfo in active_ep_buf])
            # TODO: Logging

            mean_rewards.append(rew_mean_10)
            datapoints.append([step, rew_mean_10])

            metric_logger.logkv("misc/serial_timesteps", update * ExpConfig.NUM_STEPS)
            metric_logger.logkv("misc/nupdates", update)
            metric_logger.logkv("misc/total_timesteps", update * ExpConfig.NBATCH)
            metric_logger.logkv("fps", fps)
            metric_logger.logkv("misc/time_elapsed", tnow - tfirststart)
            metric_logger.logkv("episode/length_mean", ep_len_mean)
            metric_logger.logkv("episode/rew_mean_10", rew_mean_10)

            for (lossval, lossname) in zip(lossvals, model.loss_names):
                metric_logger.logkv(f"loss/{lossname}", lossval)

            metric_logger.dumpkvs()

        if can_save:
            if update % ExpConfig.SAVE_INTERVAL == 0 or update == 1:
                misc_util.save_model(model, exp_folder_path / f"auto_save_{update}")

    misc_util.save_model(model, exp_folder_path / "last_model")
    env.close()
