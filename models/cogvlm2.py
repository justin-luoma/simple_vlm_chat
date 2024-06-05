class CogVLM2Model:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

        import torch
        import transformers
        import warnings

        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')
        torch.set_default_device("cuda")
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        if 'int4' in self.model_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": 1},
                trust_remote_code=True,
                load_in_4bit=True,
                low_cpu_mem_usage=True,
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            ).to(torch.bfloat16).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        return self.model, self.tokenizer

    def unload(self):
        import torch

        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
        torch.cuda.empty_cache()

    def generate(self, prompt, image, temperature=0.2, top_p=1.0, max_output_tokens=896, repetition_penalty=1.0):
        from transformers import TextIteratorStreamer
        import torch
        from threading import Thread

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
