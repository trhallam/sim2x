"""Provides tools for interpreting how to build a dask cluster locally, or to
connect to a remote dask cluster.
"""
import psutil

from dask.distributed import Client, LocalCluster


def get_number_real_cores():
    # get the number of real cores
    return psutil.cpu_count(logical=False)


def get_client(cluster_address: str = None, njobs: int = None) -> Client:
    """
    Args:
        cluster_address:
        njobs: Number of processes to use. Defaults to number of logical cores if local.
    """
    n_real_cores = get_number_real_cores()
    if njobs is None:
        njobs = n_real_cores
    elif njobs > n_real_cores:
        raise ValueError(f"Not enough cores for njobs={njobs}")
    else:
        pass

    if cluster_address is None:
        cluster = LocalCluster(n_workers=njobs, threads_per_worker=1)
    else:
        cluster = cluster_address

    client = Client(cluster)
    return client
