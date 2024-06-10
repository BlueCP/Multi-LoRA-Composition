import os
import torch
import argparse
from tqdm import tqdm
from os.path import join, exists
from SD.pipelines.stable_diffusion.pipeline import StableDiffusionPipeline
from diffusers import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler

from utils import load_lora_info, generate_combinations
from utils import get_prompt
from callbacks import make_callback

import numpy as np

def main(args):

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

    # set path based on the image style
    args.save_path = args.save_path + "_" + args.image_style
    args.lora_path = join(args.lora_path, args.image_style)

    # load all the information of LoRAs
    lora_info = load_lora_info(args.image_style, args.lora_info_path)

    # set base model based on the image style
    if args.image_style == 'anime':
        model_name = 'gsdf/Counterfeit-V2.5'
    else:
        model_name = 'SG161222/Realistic_Vision_V5.1_noVAE'

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        # custom_pipeline="MingZhong/StableDiffusionPipeline-with-LoRA-C",
        # custom_pipeline="./pipelines/sd1.5_0.26.3",
        # torch_dtype=torch.float16,
        use_safetensors=True
    ).to('cuda')

    # set vae
    if args.image_style == "reality":
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            # torch_dtype=torch.float16
        ).to('cuda')
        pipeline.vae = vae

    # set scheduler
    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)

    # initialize LoRAs
    for element in list(lora_info.keys()):
        for lora in lora_info[element]:
            pipeline.load_lora_weights(
                args.lora_path,
                weight_name=lora['id'] + '.safetensors',
                adapter_name=lora['id']
            )

    # generate all combinations that can be composed
    combinations = generate_combinations(lora_info, args.compos_num)
    
    # prompt initialization
    init_prompt, negative_prompt = get_prompt(args.image_style)

    timestep_similarity = [torch.empty((0)).to('cuda') for i in range(args.denoise_steps - 1)]

    # generate images for each combination based on LoRAs
    for combo in tqdm(combinations):
        cur_loras = [lora['id'] for lora in combo]

        # set prompt
        triggers = [trigger for lora in combo for trigger in lora['trigger']]
        prompt = init_prompt + ', ' + ', '.join(triggers)
        
        # set LoRAs
        if args.method == "switch":
            pipeline.set_adapters([cur_loras[0]])
            switch_callback = make_callback(args.switch_step,
                                        cur_loras)
        elif args.method == "merge":
            pipeline.set_adapters(cur_loras)
            switch_callback = None
        else:
            pipeline.set_adapters(cur_loras)
            switch_callback = None

        # generate images
        result = pipeline(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.denoise_steps,
            guidance_scale=args.cfg_scale,
            generator=args.generator,
            cross_attention_kwargs={"scale": args.lora_scale},
            callback_on_step_end=switch_callback,
            lora_composite=True if args.method == "composite" else False,
            cache_layer_id=args.cache_layer_id,
            cache_block_id=args.cache_block_id
        )

        for i in range(args.denoise_steps - 1):
            timestep_similarity[i] = torch.cat((timestep_similarity[i], result[i]), dim=0)
        
        # # save image
        # save_path = join(args.save_path, f'{args.compos_num}_elements')
        # if not exists(save_path):
        #     os.makedirs(save_path)
        # file_name = args.method + '_' + '_'.join([lora['id'] for lora in combo]) + '.png'
        # image.save(join(save_path, file_name))

    for i in range(args.denoise_steps - 1):
        timestep_similarity[i] = timestep_similarity[i].detach().cpu().numpy()
    
    results = np.array(timestep_similarity)
    np.save(f'similarity/{args.method}_{args.image_style}_{args.compos_num}.npy', results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Given LoRAs in the ComposLoRA benchmark, generate images with arbitrary combinations based on LoRAs'
    )

    # Arguments for composing LoRAs
    parser.add_argument('--compos_num', default=2,
                        help='number of elements to be combined in a single image', type=int)
    parser.add_argument('--method', default='switch',
                        choices=['merge', 'switch', 'composite'],
                        help='methods for combining LoRAs', type=str)
    parser.add_argument('--save_path', default='output',
                        help='path to save the generated image', type=str)
    parser.add_argument('--lora_path', default='models/lora',
                        help='path to store all LoRA models', type=str)
    parser.add_argument('--lora_info_path', default='lora_info.json',
                        help='path to stroe all LoRA information', type=str)
    parser.add_argument('--lora_scale', default=0.8,
                        help='scale of each LoRA when generating images', type=float)
    parser.add_argument('--switch_step', default=5,
                        help='number of steps to switch LoRA during denoising, applicable only in the switch method', type=int)

    # Arguments for generating images
    parser.add_argument('--height', default=512,
                        help='height of the generated images', type=int)
    parser.add_argument('--width', default=512,
                        help='width of the generated images', type=int)
    parser.add_argument('--denoise_steps', default=200,
                        help='number of the denoising steps', type=int)
    parser.add_argument('--cfg_scale', default=10,
                        help='scale for classifier-free guidance', type=float)
    parser.add_argument('--seed', default=111,
                        help='seed for generating images', type=int)
    parser.add_argument('--image_style', default='anime',
                        choices=['anime', 'reality'],
                        help='sytles of the generated images', type=str)

    # DeepCache arguments
    parser.add_argument('--cache_layer_id', default=0, type=int)
    parser.add_argument('--cache_block_id', default=1, type=int)

    args = parser.parse_args()
    args.generator = torch.manual_seed(args.seed)

    main(args)