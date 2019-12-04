from collections import defaultdict
from coinrun_adventure.utils import mkdir
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from coinrun_adventure.config import ExpConfig
import wandb


class KVWriter:
    def writekvs(self, kvs):
        raise NotImplementedError

    def close(self):
        pass


class LoguruOutput(KVWriter):
    def __init__(self):
        from loguru import logger

        self.logger = logger

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            self.logger.info(f"{k}: {v}")


class TensorBoardOutput(KVWriter):
    def __init__(self, logdir):
        self.logdir = (Path(logdir) / "tb").resolve()
        mkdir(self.logdir)
        self.writer = SummaryWriter(log_dir=str(self.logdir))
        self.step = 1

    def writekvs(self, kvs):

        for k, v in sorted(kvs.items()):
            self.writer.add_scalar(k, float(v), self.step)

        self.writer.flush()
        self.step += 1

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None


class WandDBOutput(KVWriter):
    def __init__(self, logdir, job_type):
        self.logdir = Path(logdir).resolve()
        self.step = 1
        mkdir(self.logdir)
        wandb.init(
            job_type=job_type,
            dir=str(self.logdir),
            config=ExpConfig.to_config_dict(),
            project=ExpConfig.PROJECT,
            tags=ExpConfig.TAGS,
            sync_tensorboard=True,
        )

    def writekvs(self, kvs):
        wandb.log(kvs, step=self.step)
        self.step += 1

    def close(self):
        pass


def get_metric_logger(**kwargs):
    if Logger._instance is None:
        Logger._instance = Logger(**kwargs)
    return Logger._instance


class Logger:
    _instance = None

    def __init__(self, folder, rank=0, job_type="training"):
        self.name2val = defaultdict(float)
        self.folder = folder
        self.rank = rank
        if self.rank == 0:
            self.output_formats = [
                LoguruOutput(),
                TensorBoardOutput(self.folder),
                WandDBOutput(self.folder, job_type),
            ]

    def logkv(self, key, val):
        self.name2val[key] = val

    def dumpkvs(self):
        if self.rank == 0:
            for fmt in self.output_formats:
                if isinstance(fmt, KVWriter):
                        fmt.writekvs(self.name2val)

        self.name2val.clear()

    def close(self):
        if self.rank == 0:
            for fmt in self.output_formats:
                fmt.close()
