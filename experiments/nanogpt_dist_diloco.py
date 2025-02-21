import torch

import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.gradient_strategy import *
from DistributedSim.demo import *

from DistributedSim.models.nanogpt import GPT, GPTConfig, GPTTrainDataset
from DistributedSim.models.dataset import *

from nanogpt import arg_parse, config_gen, gen_data

def main():
    parser = arg_parse()

    parser.add_argument("--diloco_interval", type=int, default=100)

    args = parser.parse_args()

    train_dataset, val_dataset, gpt_config = gen_data(args)

    config = config_gen(args, train_dataset, val_dataset, gpt_config)

    config.gradient_class = DiLoCoGradient
    config.gradient_config.diloco_interval = args.diloco_interval
    config.gradient_config.outer_optimizer_cls = torch.optim.SGD
    config.gradient_config.outer_optimizer_kwargs = {
        'lr': 0.7,
        'nesterov': True,
        'momentum': 0.9,
    }

    simbuilder = DistributedSimBuilder(config)

    simbuilder.execute()


if __name__ == "__main__":
    main()
