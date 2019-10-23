from collections import defaultdict
import os, numpy as np
import platform
import shutil
import subprocess
import warnings
import sys
from mpi4py import MPI


def sync_from_root(variables, comm=None):
    """
    Send the root node's parameters to every worker.
    Arguments:
      variables: all parameter variables including optimizer's
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    values = comm.bcast([var.numpy() for var in variables])
    for (var, val) in zip(variables, values):
        var.assign(val)


def gpu_count():
    """
    Count the GPUs on this machine.
    """
    if shutil.which("nvidia-smi") is None:
        return 0
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"]
    )
    return max(0, len(output.split(b"\n")) - 2)


def setup_mpi_gpus():
    """
    Set CUDA_VISIBLE_DEVICES to MPI rank if not already set
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if sys.platform == "darwin":  # This Assumes if you're on OSX you're just
            ids = []  # doing a smoke test and don't want GPUs
        else:
            lrank, _lsize = get_local_rank_size(MPI.COMM_WORLD)
            ids = [lrank]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ids))


def get_local_rank_size(comm):
    """
    Returns the rank of each process on its machine
    The processes on a given machine will be assigned ranks
        0, 1, 2, ..., N-1,
    where N is the number of processes on this machine.

    Useful if you want to assign one gpu per machine
    """
    this_node = platform.node()
    ranks_nodes = comm.allgather((comm.Get_rank(), this_node))
    node2rankssofar = defaultdict(int)
    local_rank = None
    for (rank, node) in ranks_nodes:
        if rank == comm.Get_rank():
            local_rank = node2rankssofar[node]
        node2rankssofar[node] += 1
    assert local_rank is not None
    return local_rank, node2rankssofar[this_node]


def mpi_average_comm(values, comm):
    size = comm.size

    x = np.array(values)
    buf = np.zeros_like(x)
    comm.Allreduce(x, buf, op=MPI.SUM)
    buf = buf / size

    return buf


def mpi_average_train_test(values):
    return mpi_average_comm(values, MPI.COMM_WORLD.Split(0, 0))
