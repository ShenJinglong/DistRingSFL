
import os
import argparse
import logging

from torch.distributed import rpc
from torch.distributed import init_process_group

from Node import Node

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

parser = argparse.ArgumentParser()
parser.add_argument("rank", type=int)
parser.add_argument("world_size", type=int)
parser.add_argument("--master_addr", default="192.168.3.5")
parser.add_argument("--master_port", default="29500")
parser.add_argument("--gloo_socket_ifname", default="wlan0")
parser.add_argument("--tp_socket_ifname", default="wlan0")
args = parser.parse_args()

logging.info(f"\n\
    @@ node{args.rank} | rank: {args.rank} | world size: {args.world_size}\n\
    Master Addr: {args.master_addr}\n\
    Master Port: {args.master_port}\n\
    GLOO Socket IFname: {args.gloo_socket_ifname}\n\
    TP Socket IFname: {args.tp_socket_ifname}")

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['GLOO_SOCKET_IFNAME'] = args.gloo_socket_ifname
    os.environ['TP_SOCKET_IFNAME'] = args.tp_socket_ifname
    init_process_group("gloo",
        "tcp://" + args.master_addr + ":" + args.master_port,
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