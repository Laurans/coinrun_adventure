from coinrun_adventure.config import ExpConfig
from coinrun import setup_utils


def setup(**kwargs):
    ExpConfig.merge(kwargs)
    setup_utils.setup(**kwargs)
