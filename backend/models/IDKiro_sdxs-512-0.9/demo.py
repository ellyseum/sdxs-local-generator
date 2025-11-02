import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL

repo = "IDKiro/sdxs-512-0.9"
seed = 42
weight_type = torch.float32

# Load model.
pipe = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=weight_type)
# pipe.vae = AutoencoderKL.from_pretrained("IDKiro/sdxs-512-0.9/vae_large")     # use original VAE
pipe.to("cuda")

prompt = "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour"

# Ensure using the same inference steps as the loaded model and CFG set to 0.
image = pipe(
    prompt, 
    num_inference_steps=1, 
    guidance_scale=0,
    generator=torch.Generator(device="cuda").manual_seed(seed)
).images[0]

image.save("output.png")
