
import os
import argparse
import logging

from torch.distributed import rpc
from torch.distributed import init_process_group

from fl.Node import Node as FL_Node
from ringsfl import Node as RingSFL_Node

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "22222"
GLOO_SOCKET_IFNAME = "enp3s0"
TP_SOCKET_IFNAME = "enp3s0"

parser = argparse.ArgumentParser()
parser.add_argument("rank", type=int)
parser.add_argument("world_size", type=int)
args = parser.parse_args()

logging.info(f"\n\
    @@ node{args.rank} | rank: {args.rank} | world size: {args.world_size}\n\
    Master Addr: {MASTER_ADDR}\n\
    Master Port: {MASTER_PORT}\n\
    GLOO Socket IFname: {GLOO_SOCKET_IFNAME}\n\
    TP Socket IFname: {TP_SOCKET_IFNAME}")

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
        "node" + str(args.rank),
        rank=args.rank,
        world_size=args.world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            rpc_timeout=0
        )
    )

    logging.info("RPC initialized ...")

    rpc.shutdown()