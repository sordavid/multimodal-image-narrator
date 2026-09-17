from transformers import AutoProcessor, AutoModelForImageTextToText


def make_messages(image, text):
    """Create the image and text message expected by Qwen."""
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


class ImageNarrator:
    def __init__(self, model_id):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
        )

    def describe(self, image, prompt, max_new_tokens=128):
        # Create the image + text message.
        messages = make_messages(image, prompt)
        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)


        # Ask model to generate answer
        # max_new_tokens controls how long answer can be
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
        # Remove original input tokens so we only keep the model's new answer
        generated_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        # Convert the model's numeric output back into readable text.
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens = True,
        )[0]

