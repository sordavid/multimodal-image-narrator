from multimodal_image_narrator.vision import ImageNarrator

# Model we want to download from hugging face
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# Image url
# image_source = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
IMAGE_SOURCE = "https://fastly.picsum.photos/id/28/4928/3264.jpg?hmac=GnYF-RnBUg44PFfU5pcw_Qs0ReOyStdnZ8MtQWJqTfA"

# ask model different questions
PROMPT = "Describe this image briefly"



def main():
    narrator = ImageNarrator(MODEL_ID)

    description = narrator.describe(
        image=IMAGE_SOURCE,
        prompt=PROMPT,
        max_new_tokens=128,
    )
    print(description)


if __name__ == "__main__":
    main()
