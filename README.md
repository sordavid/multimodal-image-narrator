# Multimodal Image Narrator

Multimodal Image Narrator is a Python project that uses the
[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
vision-language model to examine an image and generate a natural-language
description.

The current version accepts an image URL and a text prompt. The long-term goal
is to turn the generated description into speech, creating an image narration
tool that can both describe and read images aloud.

## Features

- Understands images using a local vision-language model
- Accepts custom questions or description prompts
- Uses Hugging Face Transformers and PyTorch
- Runs locally after the model has been downloaded

## How It Works

1. The image and prompt are formatted as a chat message.
2. `AutoProcessor` converts the image and text into model inputs.
3. Qwen3-VL generates an answer as tokens.
4. The processor converts those tokens back into readable text.

## Requirements

- Python 3.10 or newer
- Enough storage and memory to run Qwen3-VL-2B-Instruct
- An internet connection for the initial model download and remote images

The first run can take a while because the model weights must be downloaded.
Later runs reuse the cached model.

## Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/sordavid/multimodal-image-narrator.git
cd multimodal-image-narrator
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You can optionally sign in to Hugging Face for higher download rate limits:

```bash
hf auth login
```

## Run The Project

```bash
python qwen_vl.py
```

The model's description will be printed in the terminal.

## Try Another Image

Edit `image_source` and `prompt` in `qwen_vl.py`:

```python
image_source = "https://example.com/image.jpg"
prompt = "Describe this image briefly"
```

The URL must point directly to an image file. A webpage that displays an image,
such as a normal Unsplash photo page, may not work. A local image path or a
direct image URL is more reliable.

## Project Structure

```text
multimodal-image-narrator/
|-- qwen_vl.py       # Loads the model and generates an image description
|-- requirements.txt # Python dependencies
|-- .gitignore       # Files Git should not upload
`-- README.md        # Project documentation
```

## Technology

- Python
- PyTorch
- Hugging Face Transformers
- Qwen3-VL-2B-Instruct

## Roadmap

- Accept local image files through command-line arguments
- Add text-to-speech narration
- Let users choose short or detailed descriptions
- Add a simple graphical or web interface
- Save generated descriptions and audio files

## License

No license has been selected yet.
