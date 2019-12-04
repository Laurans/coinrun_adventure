import torch
import numpy as np
import os


def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)

    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=device, dtype=torch.float32)
    return x


def input_preprocessing(x, device):
    x = np.transpose(x, (0, 3, 1, 2))
    x = tensor(x, device)
    x /= 255.0
    return x


def to_np(t):
    return t.cpu().detach().numpy()


def random_seed(seed=None):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def save_model(model, update, save_path):
    torch.save(
        {
            "update": update,
            "model_state_dict": model.network.state_dict(),
            "optimizer_state_dict": model.optimizer.state_dict(),
        },
        save_path,
    )


def restore_model(model, save_path):
    checkpoint = torch.load(save_path)
    model.network.load_state_dict(checkpoint["model_state_dict"])
    model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    update = checkpoint["update"]
    return update


def parallel_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases
    torch.manual_seed(42)


def parallel_cleanup():
    torch.distributed.destroy_process_group()

