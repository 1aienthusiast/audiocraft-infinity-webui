# Audiocraft Infinity WebUI

Adds generation of songs with a length of over 30 seconds.

Adds the ability to continue songs.

Adds a seed option.

Adds ability to load locally downloaded models.

### Adds training (Thanks to chavinlo's repo https://github.com/chavinlo/musicgen_trainer)

Disables (hopefully) the gradio analytics.

## Installation
Python 3.9 is recommended.

1. Clone the repo:
`git clone https://github.com/1aienthusiast/audiocraft-infinity-webui.git`
2. Install pytorch:
`pip install 'torch>=2.0'`
3. Install the requirements:
`pip install -r requirements.txt`
4. Clone my fork of the Meta audiocraft repo and chavinlo's MusicGen trainer inside the `repositories` folder:
```
cd repositories
git clone https://github.com/1aienthusiast/audiocraft
git clone https://github.com/chavinlo/musicgen_trainer
cd ..
```
## Note!
If you already cloned the Meta audiocraft repo you have to remove it then clone the provided fork for the seed option to work.
```
cd repositories
rm -rf audiocraft/
git clone https://github.com/1aienthusiast/audiocraft
git clone https://github.com/chavinlo/musicgen_trainer
cd ..
```

## Usage
```python webui.py```

## Updating
Run `git pull` inside the root folder to update the webui, and the same command inside `repositories/audiocraft` to update audiocraft.

## Models

Meta provides 4 pre-trained models. The pre trained models are:
- `small`: 300M model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-small)
- `medium`: 1.5B model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-medium)
- `melody`: 1.5B model, text to music and text+melody to music - [🤗 Hub](https://huggingface.co/facebook/musicgen-melody)
- `large`: 3.3B model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-large)

**Needs a GPU!**

I recommend 12GB of VRAM for the large model.

## Training

### Dataset Creation

Create a folder, in it, place your audio and caption files. **They must be WAV and TXT format respectively.**

![](https://i.imgur.com/AlDlqBI.png)

**Place the folder in `training/datasets/`.**

### Important: Split your audios in 35 second chunks. Only the first 30 seconds will be processed. Audio cannot be less than 30 seconds.

In this example, segment_000.txt contains the caption "jazz music, jobim" for wav file segment_000.wav

### Options

- `dataset_path` - path to your dataset with WAV and TXT pairs.
- `model_id` - MusicGen model to use. Can be `small`/`medium`/`large`. Default: `small` - model it will be finetuned on
- `lr`: Float, learning rate. Default: `0.0001`/`1e-4`
- `epochs`: Integer, epoch count. Default: `5`
- `use_wandb`: Integer, `1` to enable wandb, `0` to disable it. Default: `0` = Disabled
- `save_step`: Integer, amount of steps to save a checkpoint. Default: None

### Models

Once training finishes, the model (and checkpoints) will be available under the `models/` directory.

![](https://i.imgur.com/Mu19EPb.png)

### Loading the finetuned models
Model gets saved to models/ as `lm_final.pt`

1) Place it in models/DIRECTORY_NAME/
2) In the Inference tab choose `custom` as the model and enter DIRECTORY_NAME into the input field. 
3) In the Inference tab choose the model it was finetuned on

## Colab

For google colab you need to use the `--share` flag.

## License
* The code in this repository is released under the AGPLv3 license as found in the [LICENSE file](LICENSE).
