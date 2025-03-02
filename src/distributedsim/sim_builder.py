import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler
from torch.multiprocessing import Queue

import os
from tqdm import tqdm
import pandas as pd
import datetime
from abc import ABC, abstractmethod
from .sim_config import *
from .train_node import *


class SimBuilder:
    '''
    SimBuilder is used to spawn processes and connect them together for distributed training.
    Spawns multiple TrainNode instances. TrainNode should be the same no matter rank topology/architecture.
    
    You can customize the train node class by setting config.train_node_class.
    '''
    def __init__(self, 
                 config: SimConfig):
        self.config = config
        # Allow customization of the train node class
        self.train_node_class = getattr(config, 'train_node_class', TrainNode)

    @abstractmethod
    def _build_connection(self):
        raise NotImplementedError

    def _process_cleanup(self):
        dist.destroy_process_group()

    def _execute(self, rank):
        self.rank = rank

        self._build_connection()

        # Use the configured train node class
        sim = self.train_node_class(self.config,
                  self.device,
                  self.rank)
        
        sim.train()

        self._process_cleanup()
        

    def execute(self):
        torch.multiprocessing.spawn(self._execute, args=(), nprocs=self.config.num_nodes, join=True)

class LocalSimBuilder(SimBuilder):
    def _build_connection(self):
        '''
        This is the default callback for setting up pytorch distributed connections.
        All ranks are assumed to be on the same machine, and device is defaulted to cpu.
        '''
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(12355 + self.config.gpu_offset - (10 if self.config.device == 'cpu' else 0))

        # initialize the process group
        if self.config.device_type == 'cuda':
            dist.init_process_group("nccl" if len(self.config.devices) == self.config.num_nodes else "gloo", 
                                    rank=self.rank, 
                                    world_size=self.config.num_nodes)
            self.device = torch.device(f"cuda:{self.config.devices[self.rank % len(self.config.devices)]}")
            torch.cuda.set_device(self.device)
        elif self.config.device_type == 'cpu':
            dist.init_process_group("gloo", 
                                    rank=self.rank, 
                                    world_size=self.config.num_nodes)
            self.device = torch.device("cpu")
        elif self.config.device_type == 'mps':
            dist.init_process_group("gloo", 
                                    rank=self.rank, 
                                    world_size=self.config.num_nodes)
            self.device = torch.device("mps")
        else:
            raise ValueError(f"Invalid device type: {self.config.device}")

        print(f"Rank {self.rank} using device {self.device}")

class DistributedSimBuilder(SimBuilder):
    """
    DistributedSimBuilder sets up distributed training over multiple physical machines,
    including those with MPS devices. In this design, one node (typically the one with rank 0)
    is designated as the master. All nodes (launched separately) must have their connection
    parameters (MASTER_ADDR, MASTER_PORT, RANK/node_rank, and WORLD_SIZE/num_nodes) provided
    via environment variables or through attributes on the SimConfig.

    To manage asynchronous startups (which may take around a minute per node), we set a generous 
    timeout in the init_process_group call and then enforce a barrier so that no training starts until 
    all nodes have connected.
    """

    def _build_connection(self):
        # Use environment variables if available; otherwise, fall back to config defaults.
        master_addr = os.environ.get("MASTER_ADDR", getattr(self.config, "master_addr", "localhost"))
        master_port = os.environ.get("MASTER_PORT", str(getattr(self.config, "master_port", 29500)))
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        # Get rank and world size from environment or from the config.
        rank = int(os.environ.get("RANK", getattr(self.config, "node_rank", 0)))
        world_size = int(os.environ.get("WORLD_SIZE", self.config.num_nodes))

        # Use a generous timeout to allow slower nodes to connect.
        timeout = getattr(self.config, "timeout", datetime.timedelta(minutes=5))

        # For MPS (and CPU) devices, the gloo backend is supported.
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=timeout
        )

        # Set the right device.
        if self.config.device_type == "mps":
            self.device = torch.device("mps")
        else:
            self.device = torch.device(self.config.device)

        print(f"Node {rank}/{world_size} connected at {master_addr}:{master_port} using device {self.device}")

        # Use a barrier to make sure that all nodes are connected before training starts.
        dist.barrier()

    def execute(self):
        """
        In a physical distributed setting each node runs independently. So instead of spawning
        multiple processes, we simply initialize the connection on the current node, create the TrainNode,
        and run training.
        """
        rank = int(os.environ.get("RANK", getattr(self.config, "node_rank", 0)))
        self.rank = rank
        self._build_connection()
        sim = self.train_node_class(self.config, self.device, self.rank)
        sim.train()
        self._process_cleanup()

class SingleSimBuilder(SimBuilder):
    def _build_connection(self):
        self.device = torch.device(self.config.device)