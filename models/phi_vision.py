import warnings

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor

from models.base_model import BaseModel


class PhiVisionModel(BaseModel):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.processor = None

    def _load(self, device=0):
        torch.cuda.set_device(int(device))
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": int(device)},
            trust_remote_code=True
        ).to(torch.bfloat16).cuda()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        return self.model, self.processor

    def unload(self):
        if self.processor is not None:
            del self.processor
            self.processor = None
        super().unload()

    def generate(
            self,
            prompt,
            image,
            temperature=1.0,
            top_p=1.0,
            max_output_tokens=896,
            repetition_penalty=1.0,
            min_p=0.95
    ):
        if self.model is None:
            print("model is not loaded")
        else:
            from threading import Thread
            from transformers import TextIteratorStreamer
            import torch

            messages = [
                {"role": "user", "content": f"<|image_1|>\n{prompt}"}
            ]

            prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(prompt, [image], return_tensors="pt").to(self.model.device)

            streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)

            generation_args = {
                "streamer": streamer,
                "max_new_tokens": max_output_tokens,
                "use_cache": True,
                "temperature": temperature,
                "do_sample": True if temperature > 0 else False,  # Set to True for sampling-based generation
                # "min_p": 0.95,  # Optionally set a minimum probability threshold
            }

            def _generate():
                with torch.inference_mode():
                    generate_ids = self.model.generate(
                        **inputs,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        **generation_args
                    )

            t = Thread(target=_generate)
            t.start()
            partial_message = ""
            for token in streamer:
                if token != '<':
                    partial_message += token
                    yield partial_message
