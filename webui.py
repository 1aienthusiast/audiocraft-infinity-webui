"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import torch
import requests
import datetime
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import wave ######
import contextlib #############

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
MODEL = None
def my_get(url, **kwargs):
    kwargs.setdefault('allow_redirects', True)
    return requests.api.request('get', 'http://127.0.0.1/', **kwargs)
original_get = requests.get
requests.get = my_get
import gradio as gr
requests.get = original_get
def load_model(version):
    print("Loading model", version)
    return MusicGen.get_pretrained(version)

def initial_generate(melody_boolean, MODEL, text, melody, msr, continue_file, duration, cf_cutoff):
    wav = None
    if continue_file:
        data_waveform, cfsr = (torchaudio.load(continue_file))
        wav = data_waveform.cuda()
        sliding_window_seconds=0
        with contextlib.closing(wave.open(continue_file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            sliding_window_seconds = frames / float(rate)

        if wav.dim() == 2:
            wav = wav[None]
        wav = wav[:, :, int(-cfsr * min(25,sliding_window_seconds,duration,cf_cutoff)):]
        new_chunk = MODEL.generate_continuation(wav, descriptions=[text], prompt_sample_rate=cfsr,progress=False)
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

def generate(model, text, melody, duration, topk, topp, temperature, cfg_coef,base_duration, sliding_window_seconds, continue_file, cf_cutoff):
    print(melody)
    final_length_seconds = duration
    descriptions = text
    global MODEL
    topk = int(topk)
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
    msr=None
    wav = None # wav shape will be [1, 1, sr * seconds]
    melody_boolean = False
    if melody:
        msr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
        print(melody.shape)
        if melody.dim() == 2:
            melody = melody[None]
        melody = melody[..., :int(msr * MODEL.lm.cfg.dataset.segment_duration)]
        melody_boolean = True
    
    if(duration > 30):
        for i in range(iterations_required):
            print(f"Generating {i + 1}/{iterations_required}")
            if i == 0:
                wav = initial_generate(melody_boolean, MODEL, text, melody, msr, continue_file,base_duration, cf_cutoff)
                wav = wav[:, :, :sr * sliding_window_seconds]
            else:
                new_chunk=None
                previous_chunk = wav[:, :, -sr * (base_duration - sliding_window_seconds):]
                print(previous_chunk)
                new_chunk = MODEL.generate_continuation(previous_chunk, descriptions=[text], prompt_sample_rate=sr,progress=False)
                print(new_chunk)
                wav = torch.cat((wav, new_chunk[:, :, -sr * sliding_window_seconds:]), dim=2)
    else:
        wav = initial_generate(melody_boolean, MODEL, text, melody, msr, continue_file, duration, cf_cutoff)

    print(f"Final length: {wav.shape[2] / sr}s")

    output = wav.detach().cpu().numpy()
    return MODEL.sample_rate, output


with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown("""# MusicGen""")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                text = gr.Text(label="Input Text", interactive=True)
                melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional) SUPPORTS MELODY ONLY", interactive=True)
                continue_file = gr.Audio(source="upload", type="filepath", label="Song to continue (optional) SUPPORTS ALL MODELS", interactive=True) 

            with gr.Row():
                model = gr.Radio(["melody", "medium", "small", "large"], label="Model", value="melody", interactive=True)
            with gr.Row():
                duration = gr.Slider(minimum=1, maximum=300, value=60, label="Duration", interactive=True)
                base_duration = gr.Slider(minimum=1, maximum=30, value=30, label="Base duration", interactive=True)
                sliding_window_seconds=gr.Slider(minimum=1, maximum=30, value=15, label="Sliding window", interactive=True)
                cf_cutoff=gr.Slider(minimum=1, maximum=30, value=15, label="Continuing song cutoff", interactive=True)
            with gr.Row():
                topk = gr.Number(label="Top-k", value=250, interactive=True)
                topp = gr.Number(label="Top-p", value=0, interactive=True)
                temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
            with gr.Row():
                submit = gr.Button("Submit")
            with gr.Row():
                output = gr.Audio(label="Generated Music", type="numpy")
            
    submit.click(generate, inputs=[model, text, melody, duration, topk, topp, temperature, cfg_coef,base_duration, sliding_window_seconds, continue_file, cf_cutoff], outputs=[output])
    gr.Examples(
        fn=generate,
        examples=[
            [
                "An 80s driving pop song with heavy drums and synth pads in the background",
                "./assets/bach.mp3",
                "melody"
            ],
            [
                "A cheerful country song with acoustic guitars",
                "./assets/bolero_ravel.mp3",
                "melody"
            ],
            [
                "90s rock song with electric guitar and heavy drums",
                None,
                "medium"
            ],
            [
                "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                "./assets/bach.mp3",
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

        Gradio analytics are disabled.
        """
    )





demo.launch()
