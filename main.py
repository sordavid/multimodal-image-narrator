from multimodal_image_narrator.vision import ImageNarrator
from multimodal_image_narrator.speech import SpeechNarrator

# Model we want to download from hugging face
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# Image url
# image_source = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
IMAGE_SOURCE = "https://picsum.photos/id/28/1024/768"

# ask model different questions
PROMPT = "Describe this image briefly"

# audio output file location
AUDIO_OUTPUT= "outputs/image-description.wav"



def main():
    narrator = ImageNarrator(MODEL_ID)

    description = narrator.describe(
        image=IMAGE_SOURCE,
        prompt=PROMPT,
        max_new_tokens=128,
    )
    print(description)

    speech_narrator = SpeechNarrator()

    audio_path = speech_narrator.save(
        text=description,
        output_path=AUDIO_OUTPUT,
    )
    print(f"Audio saved to {audio_path}")


if __name__ == "__main__":
    main()
