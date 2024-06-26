import warnings

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base_model import BaseModel


class BunnyModel(BaseModel):

    def generate(self, prompt, image, temperature=0.2, top_p=1.0, max_output_tokens=896, repetition_penalty=1.0):
        if self.model is None:
            print("model is not loaded")
        else:
            from threading import Thread
            from transformers import TextIteratorStreamer
            import torch

            image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype,
                                                                                    device=self.model.device)
            text = (f"A chat between a curious user and an artificial intelligence assistant. The assistant gives "
                    f"helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} "
                    f"ASSISTANT:")
            text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]
            input_ids = (
                torch.tensor(
                    text_chunks[0] + [-200] + text_chunks[1][1:],
                    dtype=torch.long)
                .unsqueeze(0)
                .to(self.model.device))
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            def _generate():
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids,
                        images=image_tensor,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_output_tokens,
                        repetition_penalty=repetition_penalty,
                        use_cache=True,
                        streamer=streamer,
                    )[0]
                    _ = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

            t = Thread(target=_generate)
            t.start()
            partial_message = ""
            for token in streamer:
                if token != '<':
                    partial_message += token
                    yield partial_message
