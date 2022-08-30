
import os
import argparse
import logging
import wandb

import torch
torch.set_num_threads(1)
from torch.distributed import rpc
from torch.distributed import init_process_group

from fl.Server import Server as FL_Server
from ringsfl.Server import Server as RingSFL_Server
from sl.Server import Server as SL_Server
from splitfed.Server import Server as SplitFed_Server

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
wandb.init(
    project="DistRingSFL",
    entity="sjinglong"
)
config = wandb.config

MASTER_ADDR = "10.0.0.1"
MASTER_PORT = "23333"
GLOO_SOCKET_IFNAME = "bat0"
TP_SOCKET_IFNAME = "bat0"
# MASTER_ADDR = "127.0.0.1"
# MASTER_PORT = "23333"
# GLOO_SOCKET_IFNAME = "lo"
# TP_SOCKET_IFNAME = "lo"

parser = argparse.ArgumentParser()
parser.add_argument("rank", type=int)
parser.add_argument("world_size", type=int)
parser.add_argument("prop_ratio", type=str)
args = parser.parse_args()

logging.info(f"\n\
    @@ server{args.rank} | rank: {args.rank} | world size: {args.world_size}\n\
    Master Addr: {MASTER_ADDR}\n\
    Master Port: {MASTER_PORT}\n\
    GLOO Socket IFname: {GLOO_SOCKET_IFNAME}\n\
    TP Socket IFname: {TP_SOCKET_IFNAME}\n\
    Algorithm: {config.alg}")

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    os.environ['GLOO_SOCKET_IFNAME'] = GLOO_SOCKET_IFNAME
    os.environ['TP_SOCKET_IFNAME'] = TP_SOCKET_IFNAME
    init_process_group("gloo",
        "tcp://" + MASTER_ADDR + ":" + MASTER_PORT,
        world_size=args.world_size,
        rank=args.rank,
        group_name="ringsfl"
    )
    rpc.init_rpc(
        "server" + str(args.rank),
        rank=args.rank,
        world_size=args.world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            rpc_timeout=0
        )
    )

    logging.info("RPC initialized ...")

    if config.alg == "ringsfl":
        server = RingSFL_Server(
            args.prop_ratio,
            config.model_type,
            config.dataset_name,
            config.dataset_type,
            config.dataset_blocknum,
            config.learning_rate,
            config.batch_size,
            config.local_epoch,
            config.global_round
        )
    elif config.alg == "fl":
        server = FL_Server(
            config.model_type,
            config.dataset_name,
            config.dataset_type,
            config.dataset_blocknum,
            config.learning_rate,
            config.batch_size,
            config.local_epoch,
            config.global_round
        )
    elif config.alg == "sl":
        server = SL_Server(
            config.model_type,
            config.dataset_name,
            config.dataset_type,
            config.dataset_blocknum,
            config.learning_rate,
            config.batch_size,
            config.local_epoch,
            config.global_round,
            config.cut_point
        )
    elif config.alg == "splitfed":
        server = SplitFed_Server(
            config.model_type,
            config.dataset_name,
            config.dataset_type,
            config.dataset_blocknum,
            config.learning_rate,
            config.batch_size,
            config.local_epoch,
            config.global_round,
            config.cut_point
        )
    server.train()

    rpc.shutdown()