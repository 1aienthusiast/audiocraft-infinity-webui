import contextlib
import datetime
import os
import random
import sys
import typing
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import requests
import torch
import torchaudio

from modules import shared

sys.path.insert(0, str(Path("repositories/audiocraft")))
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio
from audiocraft.models import MusicGen

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
def my_get(url, **kwargs):
    kwargs.setdefault('allow_redirects', True)
    return requests.api.request('get', 'http://127.0.0.1/', **kwargs)
original_get = requests.get
requests.get = my_get
import gradio as gr
requests.get = original_get


first_run = True
MODEL = None


def load_model(version):
    print("Loading model", version)
    return MusicGen.get_pretrained(version)


def set_seed(seed: int = 0):
    original_seed = seed
    if seed == -1:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if seed <= 0:
        seed = np.random.default_rng().integers(1, 2**32 - 1)
    assert 0 < seed < 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    return original_seed if original_seed > 0 else seed


def generate_cmelody(descriptions: typing.List[str], melody_wavs: typing.Union[torch.Tensor, typing.List[typing.Optional[torch.Tensor]]],
                     msr: int, prompt: torch.Tensor, psr: int, MODEL, progress: bool = False) -> torch.Tensor:
    if isinstance(melody_wavs, torch.Tensor):
        if melody_wavs.dim() == 2:
            melody_wavs = melody_wavs[None]
        if melody_wavs.dim() != 3:
            raise ValueError("melody_wavs should have a shape [B, C, T].")
        melody_wavs = list(melody_wavs)
    else:
        for melody in melody_wavs:
            if melody is not None:
                assert melody.dim() == 2, "one melody in the list has the wrong number of dims."

    melody_wavs = [
        convert_audio(wav, msr, MODEL.sample_rate, MODEL.audio_channels)
        if wav is not None else None
        for wav in melody_wavs]

    if prompt.dim() == 2:
        prompt = prompt[None]
    if prompt.dim() != 3:
        raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
    prompt = convert_audio(prompt, psr, MODEL.sample_rate, MODEL.audio_channels)
    if descriptions is None:
        descriptions = [None] * len(prompt)
    attributes, prompt_tokens = MusicGen._prepare_tokens_and_attributes(MODEL, descriptions=descriptions, prompt=prompt, melody_wavs=melody_wavs)
    assert prompt_tokens is not None
    return MusicGen._generate_tokens(MODEL, attributes, prompt_tokens, progress)


def initial_generate(melody_boolean, MODEL, text, melody, msr, continue_file, duration, cf_cutoff, sc_text):
    wav = None
    if continue_file:
        data_waveform, cfsr = (torchaudio.load(continue_file))
        wav = data_waveform.cuda()
        cf_len = 0
        with contextlib.closing(wave.open(continue_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            cf_len = frames / float(rate)
        if wav.dim() == 2:
            wav = wav[None]
        wav = wav[:, :, int(-cfsr * min(29, cf_len, duration - 1, cf_cutoff)):]
        new_chunk = None
        if not melody_boolean:
            if not sc_text:
                new_chunk = MODEL.generate_continuation(wav, prompt_sample_rate=cfsr, progress=False)
            else:
                new_chunk = MODEL.generate_continuation(wav, descriptions=[text], prompt_sample_rate=cfsr, progress=False)
            wav = new_chunk
        else:
            new_chunk = generate_cmelody([text], melody, msr, wav, cfsr, MODEL, progress=False)
            wav = new_chunk
    else:
        if melody_boolean:
            wav = MODEL.generate_with_chroma(
                descriptions=[text],
                melody_wavs=melody,
                melody_sample_rate=msr,
                progress=False
            )
        else:
            wav = MODEL.generate(descriptions=[text], progress=False)
    return wav


def generate(model, text, melody, duration, topk, topp, temperature, cfg_coef, base_duration,
             sliding_window_seconds, continue_file, cf_cutoff, sc_text, seed):
    # seed workaround
    global first_run
    if first_run:
        first_run = False
        d = generate(model, "A", None, 1, topk, topp, temperature, 2, base_duration,
                     sliding_window_seconds, None, cf_cutoff, sc_text, seed)

    final_length_seconds = duration
    descriptions = text
    global MODEL
    topk = int(topk)
    int_seed = int(seed)
    cur_seed = set_seed(int_seed)
    print("seed: " + str(cur_seed))
    if MODEL is None or MODEL.name != model:
        MODEL = load_model(model)
    if duration > 30:
        MODEL.set_generation_params(
            use_sampling=True,
            top_k=topk,
            top_p=topp,
            temperature=temperature,
            cfg_coef=cfg_coef,
            duration=base_duration,
        )
    else:
        MODEL.set_generation_params(
            use_sampling=True,
            top_k=topk,
            top_p=topp,
            temperature=temperature,
            cfg_coef=cfg_coef,
            duration=duration,
        )
    iterations_required = int(final_length_seconds / sliding_window_seconds)
    print(f"Iterations required: {iterations_required}")
    sr = MODEL.sample_rate
    print(f"Sample rate: {sr}")
    msr = None
    wav = None  # wav shape will be [1, 1, sr * seconds]
    melody_boolean = False
    if melody:
        msr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
        print(melody.shape)
        if melody.dim() == 2:
            melody = melody[None]
        melody = melody[..., :int(msr * MODEL.lm.cfg.dataset.segment_duration)]
        melody_boolean = True

    if (duration > 30):
        for i in range(iterations_required):
            print(f"Generating {i + 1}/{iterations_required}")
            if i == 0:
                wav = initial_generate(melody_boolean, MODEL, text, melody, msr, continue_file, base_duration, cf_cutoff, sc_text)
                wav = wav[:, :, :sr * sliding_window_seconds]
            else:
                new_chunk = None
                previous_chunk = wav[:, :, -sr * (base_duration - sliding_window_seconds):]
                if continue_file:
                    if not sc_text:
                        new_chunk = MODEL.generate_continuation(previous_chunk, prompt_sample_rate=sr, progress=False)
                    else:
                        new_chunk = MODEL.generate_continuation(previous_chunk, descriptions=[text], prompt_sample_rate=sr, progress=False)
                else:
                    new_chunk = MODEL.generate_continuation(previous_chunk, descriptions=[text], prompt_sample_rate=sr, progress=False)
                wav = torch.cat((wav, new_chunk[:, :, -sr * sliding_window_seconds:]), dim=2)
    else:
        wav = initial_generate(melody_boolean, MODEL, text, melody, msr, continue_file, duration, cf_cutoff, sc_text)

    print(f"Final length: {wav.shape[2] / sr}s")
    output = wav.detach().cpu().float()[0]
    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
        audio_write(file.name, output, MODEL.sample_rate, strategy="loudness", loudness_headroom_db=16, add_suffix=False, loudness_compressor=True)
    set_seed(-1)
    return file.name


with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown("""# MusicGen""")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                text = gr.Text(label="Input Text", interactive=True)
                melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional) SUPPORTS MELODY ONLY", interactive=True)
                continue_file = gr.Audio(source="upload", type="filepath",
                                         label="Song to continue (optional) SUPPORTS ALL MODELS", interactive=True)

            with gr.Row():
                model = gr.Radio(["melody", "medium", "small", "large"], label="Model", value="melody", interactive=True)
            with gr.Row():
                duration = gr.Slider(minimum=1, maximum=300, value=30, label="Duration", interactive=True)
                base_duration = gr.Slider(minimum=1, maximum=30, value=30, label="Base duration", interactive=True)
                sliding_window_seconds = gr.Slider(minimum=1, maximum=30, value=15, label="Sliding window", interactive=True)
                cf_cutoff = gr.Slider(minimum=1, maximum=30, value=15, label="Continuing song cutoff", interactive=True)
            with gr.Row():
                topk = gr.Number(label="Top-k", value=250, interactive=True)
                topp = gr.Number(label="Top-p", value=0, interactive=True)
                temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
            with gr.Row():
                sc_text = gr.Checkbox(label="Use text for song continuation.", value=True)
                seed = gr.Number(label="seed", value=-1, interactive=True)
            with gr.Row():
                submit = gr.Button("Submit")
            with gr.Row():
                output = gr.Audio(label="Generated Music", type="filepath")

    submit.click(generate, inputs=[model, text, melody, duration, topk, topp, temperature,
                                   cfg_coef, base_duration, sliding_window_seconds, continue_file, cf_cutoff, sc_text, seed], outputs=[output])
    gr.Examples(
        fn=generate,
        examples=[
            [
                "An 80s driving pop song with heavy drums and synth pads in the background",
                "./repositories/audiocraft/assets/bach.mp3",
                "melody"
            ],
            [
                "A cheerful country song with acoustic guitars",
                "./repositories/audiocraft/assets/bolero_ravel.mp3",
                "melody"
            ],
            [
                "90s rock song with electric guitar and heavy drums",
                None,
                "medium"
            ],
            [
                "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                "./repositories/audiocraft/assets/bach.mp3",
                "melody"
            ],
            [
                "lofi slow bpm electro chill with organic samples",
                None,
                "medium",
            ],
        ],
        inputs=[text, melody, model],
        outputs=[output]
    )
    gr.Markdown(
        """
        This is a webui for MusicGen with 30+ second generation support.

        Models
        1. Melody -- a music generation model capable of generating music condition on text and melody inputs. **Note**, you can also use text only.
        2. Small -- a 300M transformer decoder conditioned on text only.
        3. Medium -- a 1.5B transformer decoder conditioned on text only.
        4. Large -- a 3.3B transformer decoder conditioned on text only (might OOM for the longest sequences.) - recommended for continuing songs

        When the optional melody conditioning wav is provided, the model will extract
        a broad melody and try to follow it in the generated samples. Only the first chunk of the song will
        be generated with melody conditioning, the others will just continue on the first chunk.

        Base duration of 30 seconds is recommended.

        Sliding window of 10/15/20 seconds is recommended.

        When continuing songs, a continuing song cutoff of 5 seconds gives good results.

        Gradio analytics are disabled.
        """
    )


if shared.args.listen:
    demo.launch(share=shared.args.share, server_name=shared.args.listen_host or '0.0.0.0', server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch)
else:
    demo.launch(share=shared.args.share, server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch)
