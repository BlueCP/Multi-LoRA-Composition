from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained('gsdf/Counterfeit-V2.5', use_safetensors=True)

for i, down_block in enumerate(pipeline.unet.down_blocks):
    if hasattr(down_block, 'attentions'):
        print(f'Down block {i}, {len(down_block.attentions)} attention/resnet blocks.')
    else:
        print(f'Down block {i}.')

for i, up_block in enumerate(pipeline.unet.up_blocks):
    if hasattr(up_block, 'attentions'):
        print(f'Up block {i}, {len(up_block.attentions)} attention/resnet blocks.')
    else:
        print(f'Up block {i}.')