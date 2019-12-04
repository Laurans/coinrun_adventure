import datetime
from pathlib import Path
from typing import Union


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    return f"./log/{name}-{get_time_str()}"


def mkdir(path: Union[str, Path]):
    Path(path).mkdir(parents=True, exist_ok=True)
