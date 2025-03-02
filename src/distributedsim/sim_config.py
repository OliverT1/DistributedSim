from dataclasses import dataclass, field
from typing import Optional, Callable, Type
import torch

from .gradient_strategy.gradient_strategy import *

class SimConfig:
    def __init__(self,
                 num_nodes: Optional[int] = None,
                 model_class: Optional[Type[torch.nn.Module]] = None,
                 model_kwargs: dict = {},
                 train_dataset: Optional[torch.utils.data.Dataset] = None,
                 val_dataset: Optional[torch.utils.data.Dataset] = None,
                 batch_size: int = 64,
                 criterion_class: Optional[Type[torch.nn.Module]] = None,
                 criterion_kwargs: dict = {},
                 gradient_class: Optional[Type[GradientStrategy]] = None,
                 gradient_config: dict = {},
                 eval_interval: int = 10,
                 gpu_offset: int = 0,
                 device: str = 'cpu',
                 checkpoint_interval: int = 100,
                 diloco_interval: int = 1000,
                 **kwargs):
        self.num_nodes = num_nodes

        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

        self.criterion_class = criterion_class
        self.criterion_kwargs = criterion_kwargs

        self.gradient_class = gradient_class
        self.gradient_config = gradient_config
        self.eval_interval = eval_interval
        self.gpu_offset = gpu_offset
        self.device = device
        self.checkpoint_interval = checkpoint_interval

        # Allow additional kwargs to be set as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

### Example gradient config:
# gradient_config = {
#     "optimizer_class": torch.optim.SGD,
#     "optimizer_kwargs": {"lr": 0.01},
# }
