import os
from argparse import Namespace

import torch

from similarity_test import main

args = Namespace()

task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
if task_id is not None:
    task_id = int(task_id)
    args.method = ('merge', 'switch', 'composite')[task_id % 3]
    args.image_style = ('reality, anime')[(task_id // 3) % 2]
    if args.image_style == 'reality':
        args.denoise_steps = 100
        args.cfg_scale = 7
        args.height = 768
        args.width = 1024
    else:
        args.denoise_steps = 200
        args.cfg_scale = 10
        args.height = 512
        args.width = 512
    args.compos_num = (2, 3, 4, 5)[task_id // 6]

args.save_path = 'output'
args.lora_path = 'models/lora'
args.lora_info_path = 'lora_info.json'
args.lora_scale = 0.8
args.switch_step = 5
args.seed = 111
args.cache_layer_id = 0
args.cache_block_id = 1
args.generator = torch.manual_seed(args.seed)

main(args)