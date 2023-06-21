import contextlib
from datetime import datetime
import os
import random
import sys
import typing
import wave
import glob
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
import numpy as np
import requests
import torch
import torchaudio
import threading
from os.path import dirname, abspath
from modules import shared, ui
cuda = True
#check for mps
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    cuda = False

sys.path.insert(0, str(Path("repositories/audiocraft")))
sys.path.insert(0, str(Path("repositories/musicgen_trainer")))
from train import train
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


MODEL = None
current_directory = dirname(abspath(__file__))

demo = gr.Blocks(analytics_enabled=False)
output = None
gen_inputs=[]
generations = 0
files = []
audio_list_html = ""


def queue():
    global generations
    global audio_list_html
    global files
    while True:
        time.sleep(0.1)
        if(len(gen_inputs)>generations):
            global output
            n=gen_inputs[generations]
            d = generate(n[0],n[1],n[2],n[3],n[4],n[5],n[6],n[7],n[8],n[9],n[10],n[11],n[12],n[13],n[14],n[15])
            files.append(d)
            generations+=1
            files.reverse()
            files = files[0:10]
            audio_list_html = "<br>".join([
                f'''
                        <div>{os.path.splitext(os.path.basename(file))[0]}</div>
                        <audio controls style=" width : 100%;"><source src="/file={file}" type="audio/wav"></audio>
                    '''
                for file in files
            ])
            files.reverse()

def load_model(version, DIRECTORY_NAME, FINETUNED_ON):
    if version != "custom":
        print("Loading model", version)
        path=current_directory+"/models/" + version + "/"
        if os.path.exists(path):
            model = MusicGen.get_pretrained(directory=path,name=version)

        else: model = MusicGen.get_pretrained(name=version)
    else:
        finetuned_dir =current_directory + "/models/" + DIRECTORY_NAME + "/" + "lm_final.pt"
        model= MusicGen.get_pretrained(name=FINETUNED_ON)
        model.lm.load_state_dict(torch.load(finetuned_dir))
        model.name="custom"
    return model


def set_seed(seed: int = 0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if seed <= 0:
        seed = np.random.default_rng().integers(1, 2**32 - 1)
    seed = np.uint32(seed).item()
    assert 0 < seed < 2**32
    original_seed = seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return original_seed


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
        if cuda:
            wav = data_waveform.cuda()
        else:
            wav = data_waveform.mps_device()
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
             sliding_window_seconds, continue_file, cf_cutoff, sc_text, seed, directory_name,finetuned_on):
    global MODEL
    if MODEL is None or MODEL.name != model:
        MODEL = load_model(model,directory_name,finetuned_on)

    final_length_seconds = duration
    descriptions = text

    topk = int(topk)
    int_seed = int(seed)
    cur_seed = set_seed(int_seed)
    print("seed: " + str(cur_seed))

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
    now = datetime.now()
    d = dirname(abspath(__file__))
    file_name = d + "/results/" + now.strftime("%Y%m%d_%H%M%S") + "-" + str(cur_seed) + ".wav"
    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
        audio_write(file_name, output, MODEL.sample_rate, strategy="loudness", loudness_headroom_db=16,
                    add_suffix=False, loudness_compressor=True)
    print(file_name)
    set_seed(-1)
    return file_name

def add_queue(model, text, melody, duration, topk, topp, temperature, cfg_coef, base_duration,
             sliding_window_seconds, continue_file, cf_cutoff, sc_text, seed, directory_name,finetuned_on):
    gen_inputs.append([model, text, melody, duration, topk, topp, temperature, cfg_coef, base_duration,
             sliding_window_seconds, continue_file, cf_cutoff, sc_text, seed, directory_name,finetuned_on])
def get_datasets(path: str, ext: str):
    return ['None'] + glob(current_directory)

def train_local(dataset_path: str,
        model_id: str,
        lr: float,
        epochs: int,
        use_wandb: bool,
        save_step: int = None,):
    if save_step==0:
        save_step=None
    wandb : int
    if use_wandb:
        wandb=1
    else:
        wandb=0
    train(
        dataset_path=dataset_path,
        model_id=model_id,
        lr=lr,
        epochs=int(epochs),
        use_wandb=wandb,
        save_step=save_step,
    )

def get_audio_list():
    global audio_list_html
    return audio_list_html
with demo:
    with gr.Tab("Inference"):
        gr.Markdown("""# MusicGen Inference""")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True)
                    melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional) SUPPORTS MELODY ONLY", interactive=True)
                    continue_file = gr.Audio(source="upload", type="filepath",
                                             label="Song to continue (optional) SUPPORTS ALL MODELS", interactive=True)

                with gr.Row():
                    model = gr.Radio(["melody", "medium", "small", "large", "custom"], label="Model", value="small", interactive=True)
                    directory_name= gr.Text(label="Finetuned DIRECTORY_NAME", interactive=True)
                    finetuned_on = gr.Radio(["small", "medium", "large"], label="FINETUNED_ON model", value="small", interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=300, value=30,step=1, label="Duration", interactive=True)
                    base_duration = gr.Slider(minimum=1, maximum=30, value=30, step=1, label="Base duration", interactive=True)
                    sliding_window_seconds = gr.Slider(minimum=1, maximum=30, value=15, step=1, label="Sliding window", interactive=True)
                    cf_cutoff = gr.Slider(minimum=1, maximum=30, value=15, step=1, label="Continuing song cutoff", interactive=True)
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
                refresh = gr.Button("Refresh")

                with gr.Row():
                    #output = gr.Audio(label="Generated Music", type="filepath")
                    output = gr.HTML()

                refresh.click(fn=get_audio_list, inputs=[], outputs=[output])
                submit.click(add_queue, inputs=[model, text, melody, duration, topk, topp, temperature,
                                               cfg_coef, base_duration, sliding_window_seconds, continue_file, cf_cutoff, sc_text, seed,directory_name,finetuned_on])
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

                    When continuing songs, a continuing song cutoff of 5 seconds gives good results. Continuing song cutoff - number of seconds to be taken from the end of the continuing song.

                    Gradio analytics are disabled.
                    """
                )
    with gr.Tab("Training"):
        with gr.Row():
            with gr.Column():
                dataset_path = gr.Dropdown(choices=glob.glob(current_directory+"/training/datasets/*/"), value='None',
                                      label='Dataset', info='The dataset path to use for training.', interactive=True)
                ui.create_refresh_button(dataset_path, lambda: None,
                                         lambda: {'choices': glob.glob(current_directory+"/training/datasets/*/")},
                                         'refresh-button')
            with gr.Column():
                lr =  gr.Number(label="Learning rate", value=0.0001, interactive=True)
                epochs = gr.Number(label="Epoch count", value=5, interactive=True)
                use_wandb = gr.Checkbox(label="Use WanDB", value=False, interactive=True)
                save_step = gr.Number(label="Number of steps after which to save a checkpoint. 0 is treated as none.", value=0, interactive=True)
        with gr.Row():
            model_id = gr.Radio(["small", "medium", "large"], label="Model", value="small", interactive=True)
        train_button = gr.Button(label="Start training")
        train_button.click(train_local, inputs=[dataset_path,model_id, lr, epochs, use_wandb, save_step], outputs=[output])
        gr.Markdown(
            """
            # Training

            Model gets saved to models/ as `lm_final.pt`
            ### Using the finetuned model

            1) Place it in models/DIRECTORY_NAME/
            2) In the Inference tab choose `custom` as the model and enter DIRECTORY_NAME into the input field.
            3) In the Inference tab choose the model it was finetuned on

            ### Options

            - `dataset_path` path to your dataset with WAV and TXT pairs.
            - `model_id - MusicGen model to use. Can be `small`/`medium`/`large`. Default: `small` - model it will be finetuned on
            - `lr`: Float, learning rate. Default: `0.0001`/`1e-4`
            - `epochs`: Integer, epoch count. Default: `5`
            - `use_wandb`: Integer, `1` to enable wandb, `0` to disable it. Default: `0` = Disabled
            - `save_step`: Integer, amount of steps to save a checkpoint. Default: None

            Gradio analytics are disabled.
            """
        )
x = threading.Thread(target=queue)
x.start()
if shared.args.listen:
    demo.launch(share=shared.args.share, server_name=shared.args.listen_host or '0.0.0.0', server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch)
else:
    demo.launch(share=shared.args.share, server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch)
