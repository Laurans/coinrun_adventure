from mpi4py import MPI


class SingletonConfig:
    __instance = None

    def __new__(cls):
        if SingletonConfig.__instance is None:
            SingletonConfig.__instance = object.__new__(cls)
            SingletonConfig.__instance.init()
        return SingletonConfig.__instance

    def init(self):

        self.NUM_WORKERS = 1  # NOTE: Aka num_envs

        # One of {'standard', 'platform', 'maze'} (for CoinRun, CoinRun-Platforms, Random-Mazes)
        ##
        self.GAME_TYPE = "standard"

        # The number of levels in the training set.
        # If NUM_LEVELS = 0, the training set is unbounded. All level seeds will be randomly generated.
        # Use SET_SEED = -1 and NUM_LEVELS = 500 to train with the same levels in the paper.
        ##
        self.NUM_LEVELS = 0

        # Provided as a seed for training set generation.
        # If SET_SEED = -1, this seed is not used and level seeds with be drawn from the range [0, NUM_LEVELS).
        # Use SET_SEED = -1 and NUM_LEVELS = 500 to train with the same levels in the paper.
        # NOTE: This value must and will be saved, in order to use the same training set for evaluation and/or visualization.
        ##
        self.SET_SEED = -1

        # Should the agent's velocity be painted in the upper left corner of observations.
        # 1/0 means True/False
        # PAINT_VEL_INFO = -1 uses smart defaulting -- will default to 1 if GAME_TYPE is 'standard' (CoinRun), 0 otherwise
        ##
        self.PAINT_VEL_INFO = -1

        # Should data augmentation be used
        ##
        self.USE_DATA_AUGMENTATION = False

        # Should observation be transformed to grayscale
        ##
        self.USE_BLACK_WHITE = False

        # Only generate high difficulty levels
        ##
        self.HIGH_DIFFICULTY = False

        # Use high resolution images for rendering
        ##
        self.IS_HIGH_RES = False
        # if not pathlib.Path(self.WORKDIR).exists():
        #     os.makedirs(self.WORKDIR, exist_ok=True)

        self.TEST_EVAL = False
        self.TEST = False

        self.compute_args_dependencies()

    def is_test_rank(self):
        if self.TEST:
            rank = MPI.COMM_WORLD.Get_rank()
            return rank % 2 == 1

        return False

    def compute_args_dependencies(self):
        if self.PAINT_VEL_INFO < 0:
            if self.GAME_TYPE == "standard":
                self.PAINT_VEL_INFO = 1
            else:
                self.PAINT_VEL_INFO = 0

        if self.TEST_EVAL:
            self.NUM_LEVELS = 0
            self.HIGH_DIFFICULTY = 1

        self.TRAIN_TEST_COMM = MPI.COMM_WORLD.Split(1 if self.is_test_rank() else 0, 0)

    def merge(self, config_dict):
        for key in config_dict:
            if key.upper() in self.__dict__:
                setattr(self, key.upper(), config_dict[key])


Config = SingletonConfig()
