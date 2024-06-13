from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer

from models.base_model import BaseModel


class CogVLM2Model(BaseModel):

    def _load(self, device=0):
        torch.cuda.set_device(int(device))
        if 'int4' in self.model_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": int(device)},
                trust_remote_code=True,
                load_in_4bit=True,
                low_cpu_mem_usage=True,
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"", int(device)},
                trust_remote_code=True
            ).to(torch.bfloat16).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        return self.model, self.tokenizer

    def generate(self, prompt, image, temperature=0.2, top_p=1.0, max_output_tokens=896, repetition_penalty=1.0):

        if self.model is None:
            print("model is not loaded")
        else:
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=prompt,
                template_version='chat',
                images=[image]
            )
            inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.model.device),
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.model.device),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.model.device),
                'images': [[input_by_model['images'][0].to(self.model.device).to(torch.bfloat16)]],
            }
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_output_tokens,
                "do_sample": True,
                'streamer': streamer,
                'repetition_penalty': repetition_penalty,
            }

            def _generate():
                with torch.no_grad():
                    return self.model.generate(**inputs, **gen_kwargs)

            t = Thread(target=_generate)
            t.start()
            partial_message = ""
            for token in streamer:
                partial_message += token
                yield partial_message
