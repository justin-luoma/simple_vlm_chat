class BunnyModel:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    from PIL import Image
    from bunny.util.utils import disable_torch_init
    from bunny.util.mm_utils import KeywordsStoppingCriteria
    from threading import Thread

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
        torch.cuda.set_device(1)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).to(torch.bfloat16).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
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
        if self.model is None:
            print("model is not loaded")
        else:
            from bunny.util.mm_utils import KeywordsStoppingCriteria
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
