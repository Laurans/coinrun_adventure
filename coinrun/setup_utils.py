from coinrun.config import Config


def setup(**kwargs):
    Config.merge(kwargs)
    from coinrun.coinrunenv import init_args_and_threads

    init_args_and_threads()
