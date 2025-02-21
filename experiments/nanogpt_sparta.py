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

def gen_wandb_name(args):
    name = f"p{args.p_sparta}_n{args.num_nodes}_lr{args.learning_rate:.0e}"
    return name

def main():
    parser = arg_parse()

    parser.add_argument("--p_sparta", type=float, default=0.005)

    args = parser.parse_args()

    train_dataset, val_dataset, gpt_config = gen_data(args)

    config = config_gen(args, train_dataset, val_dataset, gpt_config)

    config.gradient_class = SPARTAGradient
    config.gradient_config.p_sparta = args.p_sparta

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()


if __name__ == "__main__":
    main()