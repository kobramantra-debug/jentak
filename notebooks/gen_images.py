# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "accelerate==1.15.0",
#     "av==19.0.1",
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


@app.cell
def _(torch):
    from diffusers import LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    ltx_model_id = "Lightricks/LTX-2.5-Diffusers"
    ltx_height, ltx_width, ltx_frames, ltx_fps = 512, 512, 121, 24.0

    ltx_pipe = LTX2ImageToVideoPipeline.from_pretrained(ltx_model_id, dtype=torch.bfloat16)
    ltx_pipe.enable_model_cpu_offload()
    ltx_pipe.vae.enable_tiling()

    ltx_upsampler = LTX2LatentUpsamplerModel.from_pretrained(ltx_model_id, subfolder="latent_upsampler", dtype=torch.bfloat16).to("cuda")
    ltx_up_pipe = LTX2LatentUpsamplePipeline(vae=ltx_pipe.vae, latent_upsampler=ltx_upsampler)

    ltx_first_frame = load_image("outputs/t2i_test.png")
    ltx_shared = dict(
        image=ltx_first_frame,
        prompt="The astronaut slowly turns their head toward the camera, leaves rustle in the jungle wind, subtle camera dolly-in, cinematic lighting",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        frame_rate=ltx_fps,
        guidance_scale=1.0,
        audio_guidance_scale=1.0,
        stg_scale=0.0,
        audio_stg_scale=0.0,
        modality_scale=1.0,
        audio_modality_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(42),
        return_dict=False,
    )

    ltx_latents, ltx_audio_latents = ltx_pipe(
        height=ltx_height, width=ltx_width, num_frames=ltx_frames,
        sigmas=DISTILLED_SIGMA_VALUES, output_type="latent", **ltx_shared,
    )
    ltx_up_latents = ltx_up_pipe(latents=ltx_latents, output_type="latent", return_dict=False)[0]
    ltx_video, ltx_audio = ltx_pipe(
        num_frames=ltx_frames,
        sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
        latents=ltx_up_latents,
        audio_latents=ltx_audio_latents,
        noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
        output_type="np",
        **ltx_shared,
    )
    encode_video(
        ltx_video[0], fps=int(ltx_fps), output_path="outputs/i2v_test.mp4",
        audio=ltx_audio[0].float().cpu(),
        audio_sample_rate=ltx_pipe.vocoder.config.output_sampling_rate,
    )
    print("saved outputs/i2v_test.mp4")

    return


if __name__ == "__main__":
    app.run()
