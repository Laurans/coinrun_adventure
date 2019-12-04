from collections import defaultdict
from coinrun_adventure.utils import mkdir
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


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


def get_metric_logger(**kwargs):
    if Logger._instance is None:
        Logger._instance = Logger(**kwargs)
    return Logger._instance


class Logger:
    _instance = None

    def __init__(self, folder, output_formats=None, rank=0):
        self.name2val = defaultdict(float)
        self.folder = folder
        self.rank = rank
        if output_formats is None:
            self.output_formats = [LoguruOutput(), TensorBoardOutput(self.folder)]
        else:
            self.output_formats = output_formats

    def logkv(self, key, val):
        self.name2val[key] = val

    def dumpkvs(self):

        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                if self.rank == 0:
                    fmt.writekvs(self.name2val)

        self.name2val.clear()

    def close(self):
        for fmt in self.output_formats:
            fmt.close()
