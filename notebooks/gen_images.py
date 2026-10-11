# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "accelerate==1.15.0",
#     "diffusers==0.41.0.dev0",
#     "ftfy==6.3.1",
#     "huggingface-hub==1.33.0",
#     "imageio==2.38.1",
#     "imageio-ffmpeg==0.6.0",
#     "pillow==12.3.0",
#     "safetensors==0.8.0",
#     "sentencepiece==0.2.2",
#     "torch==2.14.1",
#     "transformers>=5.17",
# ]
#
# [tool.uv.sources]
# diffusers = { git = "https://github.com/huggingface/diffusers" }
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import torch


    return (torch,)


@app.cell
def _(torch):
    _props = torch.cuda.get_device_properties(0)
    print(f"GPU: {_props.name} | VRAM: {_props.total_memory / 1e9:.1f} GB | torch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    return


@app.cell
def _(torch):
    import os
    from diffusers import QwenImage21Pipeline

    os.makedirs("outputs", exist_ok=True)

    t2i_pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
    )
    t2i_pipe.enable_model_cpu_offload()

    t2i_image = t2i_pipe(
        prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        width=1024,
        height=1024,
        num_inference_steps=30,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    t2i_image

    return (t2i_image,)


@app.cell
def _(t2i_image):
    import numpy as _np
    import imageio.v2 as _imageio

    _imageio.imwrite("outputs/t2i_test.png", _np.array(t2i_image))
    print("saved outputs/t2i_test.png", t2i_image.size)

    return


if __name__ == "__main__":
    app.run()
