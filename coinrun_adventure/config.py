from coinrun.config import Config
from pathlib import Path


def constfn(val):
    def f(_):
        return val

    return f


class SingletonExpConfig:
    __instance = None

    def __new__(cls):
        if SingletonExpConfig.__instance is None:
            SingletonExpConfig.__instance = object.__new__(cls)
            SingletonExpConfig.__instance.init()
        return SingletonExpConfig.__instance

    def init(self):
        self.PROJECT = "coinrun"
        self.TAGS = ["PPO"]

        self.WORLD_SIZE = 2

        self.ENV_CONFIG = Config
        self.DEVICE = "cuda"

        self.SAVE_DIR = Path(__file__).parent.parent.joinpath("experiment_results")

        self.NUM_ENVS = 32 * (8 // self.WORLD_SIZE)

        # Number of timesteps i.e. number of actions taken in the environment
        self.TOTAL_TIMESTEPS = 256e6

        self.ENTROPY_WEIGHT = (
            0.01
        )  # Policy entropy coefficient in the optimization objective
        self.LEARNING_RATE = 5e-4  # Learning rate, constant
        self.LR_FN = lambda f: f * self.LEARNING_RATE
        self.VALUE_WEIGHT = (
            0.5
        )  # Value function loss coefficient in the optimization objective
        self.MAX_GRAD_NORM = 0.5  # Gradient norm clipping coefficient
        self.GAMMA = 0.999  # discounting factor
        self.LAMBDA = 0.95  # advantage estimation discounting factor
        # number of training minibatches per update. For recurrent policies should be small of
        # equal than number of environment run in parallel.
        self.NUM_MINI_BATCH = 8
        self.NUM_OPT_EPOCHS = 3  # number of training epochs per update
        self.NUM_STEPS = 256  # NOTE: rollout length
        self.CLIP_RANGE = 0.2  # clipping range, constant
        self.CLIP_RANGE_FN = lambda f: f * self.CLIP_RANGE

        self.LOG_INTERVAL = 5  # Number of updates betwen logging events

        # The convolutional architecture to use
        # One of {'NatureConv', 'impala', 'impalalarge'}
        self.ARCHITECTURE = "NatureConv"

        # Should the model include an LSTM
        self.USE_LSTM = False

        # Should batch normalization be used after each convolutional layer
        # NOTE: Only applies to IMPALA and IMPALA-Large architectures
        self.USE_BATCH_NORM = True

        # What dropout probability to use after each convolutional layer
        # NOTE: Only applies to IMPALA and IMPALA-Large architectures
        self.DROPOUT = 0.0

        # The L2 penalty to use during training
        self.L2_WEIGHT = 1e-4

        # The probability the agent's action is replaced with a random action
        self.EPSILON_GREEDY = 0.0

        # The number of frames to stack for each observation.
        self.FRAME_STACK = 1

        # Overwrite the latest save file after this many updates
        self.SAVE_INTERVAL = 100

        self.compute_args_dependencies()

    def compute_args_dependencies(self):
        self.merge(self.ENV_CONFIG.__dict__)

        self.NBATCH = self.NUM_ENVS * self.NUM_STEPS
        self.NBATCH_TRAIN = self.NBATCH // self.NUM_MINI_BATCH

        self.TAGS += [self.ARCHITECTURE]

    def merge(self, config_dict):
        for key in config_dict:
            setattr(self, key.upper(), config_dict[key])

    def to_config_dict(self):
        config = {}
        for k, v in self.__dict__.items():
            if type(v) in [int, float, str, bool, tuple, list]:
                config[k] = v

        return config


ExpConfig = SingletonExpConfig()
