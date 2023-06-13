# Audiocraft Infinity WebUI

Adds generation of songs with a length of over 30 seconds.

Adds the ability to continue songs.

Adds a seed.

Disables (hopefully) the gradio analytics.

## Installation
1. Clone the repo:
`git clone https://github.com/1aienthusiast/audiocraft-infinity-webui.git`
2. Install the requirements:
`pip install -r requirements.txt`
3. Clone the Meta audiocraft repo:
`git clone https://github.com/facebookresearch/audiocraft`
4. Copy the webui.py to the directory of the Meta audiocraft repo
`cp webui.py ./audiocraft/`
5. Go to the audiocraft directory
`cd ./audiocraft/`
## Usage
```python webui.py```
## Models

Meta provides 4 pre-trained models. The pre trained models are:
- `small`: 300M model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-small)
- `medium`: 1.5B model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-medium)
- `melody`: 1.5B model, text to music and text+melody to music - [🤗 Hub](https://huggingface.co/facebook/musicgen-melody)
- `large`: 3.3B model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-large)

**Needs a GPU!**

I recommend 12GB of VRAM for the large model.

## Colab

For google colab you need to replace `demo.launch()` with `demo.queue().launch(share=True)` in webui.py

## License
* The code in this repository is released under the AGPLv3 license as found in the [LICENSE file](LICENSE).
