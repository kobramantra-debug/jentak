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
    from diffusers import QwenImage21Pipeline as _QwenPipe

    t2i_real_pipe = _QwenPipe.from_pretrained(
        "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
    )
    t2i_real_pipe.enable_model_cpu_offload()

    t2i_real_image = t2i_real_pipe(
        prompt="Photorealistic close-up portrait of an astronaut in a worn white spacesuit with scratched visor, standing in a dense jungle, natural skin texture with sweat droplets, volumetric god rays through canopy leaves, shallow depth of field, shot on 85mm lens, muted cold color palette, ultra-detailed, 8k",
        width=2048,
        height=2048,
        num_inference_steps=40,
        generator=torch.Generator("cuda").manual_seed(123),
    ).images[0]
    t2i_real_image

    return (t2i_real_image,)


@app.cell
def _(t2i_real_image):
    import numpy as _np
    import imageio.v2 as _imageio

    _imageio.imwrite("outputs/t2i_real.png", _np.array(t2i_real_image))
    print("saved outputs/t2i_real.png", t2i_real_image.size)

    return


@app.cell
def _(torch):
    from diffusers import LTX2ImageToVideoPipeline as _L2P
    from diffusers import LTX2LatentUpsamplePipeline as _L2U
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel as _L2M

    ltx2_pipe = _L2P.from_pretrained("Lightricks/LTX-2.5-Diffusers", dtype=torch.bfloat16)
    ltx2_pipe.enable_model_cpu_offload()
    ltx2_pipe.vae.enable_tiling()

    ltx2_up = _L2M.from_pretrained("Lightricks/LTX-2.5-Diffusers", subfolder="latent_upsampler", dtype=torch.bfloat16).to("cuda")
    ltx2_up_pipe = _L2U(vae=ltx2_pipe.vae, latent_upsampler=ltx2_up)
    print("LTX-2.5 loaded")

    return ltx2_pipe, ltx2_up_pipe


@app.cell
def _(ltx2_pipe, ltx2_up_pipe, torch):
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT as _DefNeg
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES as _Sig1
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES as _Sig2
    from diffusers.utils import load_image as _load_img
    from diffusers.utils.export_utils import encode_video as _enc_vid

    real_frame = _load_img("outputs/t2i_real.png")
    real_shared = dict(
        image=real_frame,
        prompt="A close-up frames the astronaut's scratched visor reflecting the jungle canopy. The astronaut breathes slowly, fog blooming and fading on the inside of the visor, and turns their head toward the camera. Leaves tremble and fall through volumetric god rays as the camera pushes in with small amplitude at slow speed. Distant bird calls and rustling foliage fill the humid air, underscored by a low ambient drone.",
        negative_prompt=_DefNeg,
        frame_rate=24.0,
        guidance_scale=1.0,
        audio_guidance_scale=1.0,
        stg_scale=0.0,
        audio_stg_scale=0.0,
        modality_scale=1.0,
        audio_modality_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(123),
        return_dict=False,
    )

    real_lat, real_audio_lat = ltx2_pipe(
        height=512, width=512, num_frames=121,
        sigmas=_Sig1, output_type="latent", **real_shared,
    )
    real_up_lat = ltx2_up_pipe(latents=real_lat, output_type="latent", return_dict=False)[0]
    real_video, real_audio = ltx2_pipe(
        num_frames=121,
        sigmas=_Sig2,
        latents=real_up_lat,
        audio_latents=real_audio_lat,
        noise_scale=_Sig2[0],
        output_type="np",
        **real_shared,
    )
    _enc_vid(
        real_video[0], fps=24, output_path="outputs/i2v_real.mp4",
        audio=real_audio[0].float().cpu(),
        audio_sample_rate=ltx2_pipe.vocoder.config.output_sampling_rate,
    )
    print("saved outputs/i2v_real.mp4")

    return


if __name__ == "__main__":
    app.run()
