import torch
import torch.distributed as dist
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


def sync_initial_weights(model, rank, world_size):
    for param in model.parameters():
        if rank == 0:
            for sibling in range(1, world_size):
                print(f"sibling {sibling}")
                dist.send(param.data, dst=sibling)

        else:
            dist.recv(param.data, src=0)


def sync_gradients(model):
    world_size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size


def cleanup():
    dist.destroy_process_group()
