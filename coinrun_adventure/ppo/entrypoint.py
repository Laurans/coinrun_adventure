from coinrun_adventure.utils import setup_util, misc_util
from coinrun_adventure.config import ExpConfig
from coinrun import make
from coinrun_adventure.ppo.model import Model
from coinrun_adventure.ppo.agent import PPORunner
import time
import numpy as np
import tensorflow as tf
from coinrun_adventure.logger import get_metric_logger, Logger
from pathlib import Path
from coinrun.common.vec_env import VecEnv
from loguru import logger as log

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def run_update(update: int, nupdates: int, runner: PPORunner, model: Model):
    # Start timer
    frac = 1.0 - (update - 1.0) / nupdates

    # Calculate the learning rate
    lrnow = ExpConfig.LR_FN(frac)
    cliprangenow = ExpConfig.CLIP_RANGE_FN(frac)

    # Get minibatch
    obs, returns, masks, actions, values, neglogpacs = runner.run()

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
    return lossvals


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
    )

    return model


def learn(exp_folder_path: Path):
    metric_logger: Logger = get_metric_logger(folder=exp_folder_path)
    setup_util.setup()
    env: VecEnv = make("standard", num_envs=ExpConfig.NUM_ENVS)

    is_mpi_root = MPI is None or MPI.COMM_WORLD.Get_rank() == 0

    model: Model = get_model()

    runner: PPORunner = PPORunner(
        env=env,
        model=model,
        num_steps=ExpConfig.NUM_STEPS,
        gamma_coef=ExpConfig.GAMMA,
        lambda_coef=ExpConfig.LAMBDA,
    )

    tfirststart = time.perf_counter()

    nupdates = ExpConfig.TOTAL_TIMESTEPS // ExpConfig.NBATCH
    for update in range(1, nupdates + 1):
        assert ExpConfig.NBATCH % ExpConfig.NUM_MINI_BATCH == 0
        # Start timer
        tstart = time.perf_counter()

        if update % ExpConfig.LOG_INTERVAL == 0 or is_mpi_root:
            log.info("Stepping environment...")

        # Run an update
        lossvals = run_update(update, nupdates, runner, model)

        # End timer
        tnow = time.perf_counter()

        # Calculate the fps
        fps = int(ExpConfig.NBATCH / (tnow - tstart))

        if update % ExpConfig.LOG_INTERVAL == 0 or update == 1:
            # todo: logging
            metric_logger.logkv("misc/serial_timesteps", update * ExpConfig.NUM_STEPS)
            metric_logger.logkv("misc/nupdates", update)
            metric_logger.logkv("misc/total_timesteps", update * ExpConfig.NBATCH)
            metric_logger.logkv("fps", fps)
            metric_logger.logkv("misc/time_elapsed", tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                metric_logger.logkv(f"loss/{lossname}", lossval)

            metric_logger.dumpkvs()

        if update % ExpConfig.SAVE_INTERVAL == 0 or update == 1:
            misc_util.save_model(model, exp_folder_path / "auto_save")

    misc_util.save_model(model, exp_folder_path / "last_model")
