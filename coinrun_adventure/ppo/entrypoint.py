from coinrun_adventure.utils.torch_utils import (
    save_model,
    sync_initial_weights,
    sync_values,
    to_np,
    tensor,
)
from coinrun_adventure.config import ExpConfig
from coinrun_adventure.ppo.model import Model
from coinrun_adventure.ppo.runner import Runner
import time
import numpy as np
from coinrun_adventure.logger import get_metric_logger, Logger
from pathlib import Path
from loguru import logger as logo
from collections import deque
import datetime


def process_ep_buf(epinfobuf, device, key):
    list_values = [epinfo[key] for epinfo in epinfobuf]

    tensor_mean = tensor(np.nanmean(list_values), device)
    value_mean = to_np(sync_values(tensor_mean))

    return value_mean


def run_update(update: int, nupdates: int, runner: Runner, model: Model):
    frac = 1.0 - (update - 1.0) / nupdates

    # Calculate the learning rate
    lrnow = ExpConfig.LR_FN(frac)
    cliprangenow = ExpConfig.CLIP_RANGE_FN(frac)

    # Get minibatch
    data_sampled, epinfos = runner.run()

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
            slices = {key: data_sampled[key][mbinds] for key in data_sampled}
            mblossvals.append(model.train(lrnow, cliprangenow, slices))

    # Feedforward --> get losses --> updates
    lossvals = np.mean(mblossvals, axis=0)
    return lossvals, epinfos


def get_model(device):
    model = Model(
        ob_shape=ExpConfig.OB_SHAPE,
        ac_space=ExpConfig.AC_SPACE.n,
        policy_network_archi=ExpConfig.ARCHITECTURE,
        ent_coef=ExpConfig.ENTROPY_WEIGHT,
        vf_coef=ExpConfig.VALUE_WEIGHT,
        l2_coef=ExpConfig.L2_WEIGHT,
        max_grad_norm=ExpConfig.MAX_GRAD_NORM,
        device=device,
    )

    return model


def learn(rank: int, exp_folder_path: Path, env):
    metric_logger: Logger = get_metric_logger(folder=exp_folder_path, rank=rank)
    device = f"{ExpConfig.DEVICE}:{rank}"
    model: Model = get_model(device)

    sync_initial_weights(model.network)

    runner = Runner(
        env=env,
        model=model,
        num_steps=ExpConfig.NUM_STEPS,
        gamma_coef=ExpConfig.GAMMA,
        lambda_coef=ExpConfig.LAMBDA,
        device=device,
    )

    epinfobuf10 = deque(maxlen=10)
    epinfobuf100 = deque(maxlen=100)

    tfirststart = time.perf_counter()

    nupdates = int(ExpConfig.TOTAL_TIMESTEPS // ExpConfig.NBATCH)

    for update in range(1, nupdates + 1):
        if rank == 0:
            logo.info(f"{update}/{nupdates+1}")
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
            rew_mean_100 = process_ep_buf(epinfobuf100, device, "r")
            rew_mean_10 = process_ep_buf(epinfobuf10, device, "r")
            ep_len_mean = process_ep_buf(epinfobuf100, device, "l")

            time_elapsed = tnow - tfirststart
            completion_perc = update * ExpConfig.NBATCH / ExpConfig.TOTAL_TIMESTEPS
            time_remaining = datetime.timedelta(
                seconds=(time_elapsed / completion_perc - time_elapsed)
            )

            metric_logger.logkv("misc/iter_update", update)
            metric_logger.logkv("misc/total_timesteps", update * ExpConfig.NBATCH)
            metric_logger.logkv("fps", fps)
            metric_logger.logkv("misc/time_elapsed", time_elapsed)
            metric_logger.logkv("episode/length_mean_100", ep_len_mean)
            metric_logger.logkv("episode/rew_mean_100", rew_mean_100)
            metric_logger.logkv("episode/rew_mean_10", rew_mean_10)
            metric_logger.logkv("misc/completion_training", completion_perc)
            logo.info(f"Time remaining {time_remaining}")

            for (lossval, lossname) in zip(lossvals, model.loss_names):
                metric_logger.logkv(f"loss/{lossname}", lossval)

            metric_logger.dumpkvs()

        if rank == 0 and (update % ExpConfig.SAVE_INTERVAL == 0 or update == 1):
            save_model(model, update, exp_folder_path / f"auto_save_{update}")

    if rank == 0:
        save_model(model, update, exp_folder_path / "last_model")

    env.close()
    metric_logger.close()
