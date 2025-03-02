import math
import torch
import torch.distributed as dist
from torch import nn

from .gradient_strategy import GradientStrategy
from .communicate import *

class SPARTAGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)

        self.optim = self.gradient_config.optimizer_class(model.parameters(), 
                                                          **self.gradient_config.optimizer_kwargs)
        self._setup_scheduler()

        # self.index_selector = PartitionedIndexSelector(self.gradient_config.p_sparta)
        self.index_selector = RandomIndexSelector(self.gradient_config.p_sparta)
        # self.buffer = []

    def step(self):
        if self.gradient_config.max_norm:
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_config.max_norm)
            # print(f'Rank {self.rank}: Clipped grad norm to {norm}')

        self.optim.step()

        if self.config.num_nodes > 1:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue

                    indices = self.index_selector.get_indices(param)
                    broadcast(indices, src=0)
                    sparse_data = param.data[indices]
                    all_reduce(sparse_data, op=dist.ReduceOp.SUM)
                    sparse_data /= dist.get_world_size()

                    param.masked_scatter_(indices, sparse_data)

                    # self.buffer.append((indices, sparse_data))
                    # if len(self.buffer) > self.gradient_config.async_sparta_delay:
                        # indices_popped, sparse_data_popped = self.buffer.pop(0)
                        # param.masked_scatter_(indices_popped, sparse_data_popped)

        super().step()

class IndexSelector:
    def __init__(self, p):
        self.state = {}
        self.p = p

    def get_indices(self, param):
        return torch.ones(param.shape).bool()


class RandomIndexSelector(IndexSelector):
    def get_indices(self, param):
        return torch.bernoulli(torch.full(param.shape, self.p, device=param.device)).bool()


class PartitionedIndexSelector(IndexSelector):
    def __init__(self, p):
        super().__init__(p)

    def _set_partition(self, param):
        param_state = self.state[param]
        param_state["curr_partition"] = 0
        param_state["num_partitions"] = min(math.ceil(1 / self.p), param.numel())
        param_state["partitions"] = (
            torch.rand(param.numel(), device=param.device).argsort().view(param.shape) % param_state["num_partitions"]
        )

    def get_indices(self, param):
        if param not in self.state:
            self.state[param] = {}
            self._set_partition(param)
        elif self.state[param]["curr_partition"] >= self.state[param]["num_partitions"]:
            self._set_partition(param)

        indices = (self.state[param]["partitions"] == self.state[param]["curr_partition"]).bool()

        self.state[param]["curr_partition"] += 1

        return indices
