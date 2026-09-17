from transformers import AutoProcessor, AutoModelForImageTextToText

# Model we want to download from hugging face
model_id = "Qwen/Qwen3-VL-2B-Instruct"

# Image url
# image_source = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
image_source = "https://fastly.picsum.photos/id/28/4928/3264.jpg?hmac=GnYF-RnBUg44PFfU5pcw_Qs0ReOyStdnZ8MtQWJqTfA"

# ask model different questions
prompt = "Describe this image briefly"

# Function to create the image + text message.
def make_messages(image, text):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": text,
                },
            ],
        }
    ]

# Load the AI model
# dtype ="auto" lets transformers choose good number format
# device_map="auto" lets Transformers decide whether to use CPU/GPU acceleration.
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)

# Load the processor
# processor prepares images and text so the model can understand them.
processor = AutoProcessor.from_pretrained(model_id)

# Create the image + text message.
messages = make_messages(image_source, prompt)

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# Ask model to generate answer
# max_new_tokens controls how long answer can be
output_ids = model.generate(
    **inputs,
    max_new_tokens=128,
)

# Remove original input tokens so we only keep the model's new answer
generated_ids = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
]

# Convert the model's numeric output back into readable text.
answer = processor.batch_decode(
    generated_ids,
    skip_special_tokens = True,
)[0]
print(answer)