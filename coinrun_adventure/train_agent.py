from coinrun_adventure.agents.networks import get_network_builder
from coinrun_adventure import setup_utils
from coinrun_adventure.config import ExpConfig
from coinrun import make
from coinrun_adventure.agents.model import Model
from coinrun_adventure.agents.ppo import PPOAgent
import time
from loguru import logger
import numpy as np
import tensorflow as tf


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def run_update(update: int, nupdates: int, runner: PPOAgent, model: Model, tfirststart):
    # Start timer
    tstart = time.perf_counter()
    frac = 1.0 - (update - 1.0) / nupdates

    # Calculate the learning rate
    lrnow = ExpConfig.LR_FN(frac)
    cliprangenow = ExpConfig.CLIP_RANGE_FN(frac)

    if update % ExpConfig.LOG_INTERVAL == 0:
        logger.info("Stepping environment...")

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
    # End timer
    tnow = time.perf_counter()

    # Calculate the fps
    fps = int(ExpConfig.NBATCH / (tnow - tstart))

    if update % ExpConfig.LOG_INTERVAL == 0 or update == 1:
        # TODO: Logging
        logger.info(f"misc/serial_timesteps {update*ExpConfig.NUM_STEPS}")
        logger.info(f"misc/nupdates {update}")
        logger.info(f"misc/total_timesteps {update*ExpConfig.NBATCH}")
        logger.info(f"fps {fps}")
        logger.info(f"misc/time_elapsed {tnow - tfirststart}")
        for (lossval, lossname) in zip(lossvals, model.loss_names):
            logger.info(f"loss/{lossname} {lossval}")


def main():

    setup_utils.setup()
    env = make("standard", num_envs=ExpConfig.NUM_ENVS)

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    policy_network_fn = get_network_builder(ExpConfig.ARCHITECTURE)()
    network = policy_network_fn(ob_space.shape)

    model_fn = Model

    model = model_fn(
        ac_space=ac_space,
        policy_network=network,
        value_network=None,
        ent_coef=ExpConfig.ENTROPY_WEIGHT,
        vf_coef=ExpConfig.VALUE_WEIGHT,
        max_grad_norm=ExpConfig.MAX_GRAD_NORM,
    )

    runner = PPOAgent(
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
        run_update(update, nupdates, runner, model, tfirststart)

    # Calculate the batchsize


if __name__ == "__main__":
    main()
