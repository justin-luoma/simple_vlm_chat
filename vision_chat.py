import argparse
import logging as logger


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


def load_model(state, model_name):
    state["model_name"] = model_name
    if 'bunny' in model_name.lower():
        print(f"Loading: {model_name}")
        bunny = BunnyModel(f"{state['models_path']}/{model_name}")
        bunny.load()
        state["model"] = bunny
    elif 'cogvlm2' in model_name.lower():
        print(f"Loading: {model_name}")
        cogvlm2 = CogVLM2Model(f"{state['models_path']}/{model_name}")
        cogvlm2.load()
        state["model"] = cogvlm2
    return state


def unload_model(state):
    if "model" in state and state["model"] is not None:
        print(f"unload: {state['model_name']}")
        model = state["model"]
        model.unload()
        state["model_name"] = None
        state["model"] = None
    return state


def generate(prompt, history, state):
    # pass
    if "model" in state and state["model"] is not None and "files" in prompt and len(prompt["files"]) > 0:
        from PIL import Image
        model = state["model"]
        image = Image.open(prompt["files"][0]).convert("RGB")
        for t in model.generate(prompt["text"], image, temperature=0.2, top_p=1, max_output_tokens=896,
                                repetition_penalty=1.0):
            yield t
    else:
        return ""


import time
import gradio as gr
import os


def get_models_list(models_path):
    try:
        models = os.listdir(models_path)
        models.sort()
        return models
    except:
        return []


def load_ui(state, models_path):
    state["models_path"] = models_path
    return state, gr.Dropdown(choices=get_models_list(models_path), interactive=True)


def models_path_update(state, models_path):
    state["models_path"] = models_path
    return state, gr.Dropdown(choices=get_models_list(models_path), interactive=True)


def model_selection_update(state, model_name):
    state["model_name"] = model_name
    return state


def state_change(state):
    if "model" in state and state["model"] is not None:
        return (state, gr.Button(
            visible=True,
            variant="primary"
        ),
                gr.Button(visible=True, variant="stop",
                          interactive=True))
    else:
        return state, gr.Button(), gr.Button(interactive=False)


def build_ui(models_path):
    models_path_default = models_path

    ui = gr.Blocks(
        title="Vision",
        theme=gr.themes.Default(primary_hue="violet", secondary_hue="lime"),
        analytics_enabled=False
    )
    with ui:
        state = gr.State({})

        with gr.Column():
            with gr.Row():
                with gr.Column(scale=2):
                    chat = gr.ChatInterface(
                        generate,
                        additional_inputs=[state],
                        retry_btn=None,
                        undo_btn=None,
                        multimodal=True,
                        chatbot=gr.Chatbot(
                            elem_id="chatbot",
                            height=700,
                            render=False,
                        ),
                    )
        with gr.Column():
            with gr.Row():
                model_selection = gr.Dropdown(label="Model", choices=[])
            with gr.Row():
                load_btn = gr.Button(value="Load Model", variant="primary")
                unload_btn = gr.Button(value="Unload Model", interactive=False)
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Parameters", open=False) as params_row:
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.2,
                            step=0.1,
                            interactive=True,
                            label="Temperature",
                        )
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            interactive=True,
                            label="Top P",
                        )
                        max_output_tokens = gr.Slider(
                            minimum=0,
                            maximum=1024,
                            value=512,
                            step=64,
                            interactive=True,
                            label="Max output tokens",
                        )
                        repetition_penalty = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.08,
                            step=0.01,
                            interactive=True,
                            label="Repetition penalty",
                        )
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Settings", open=False) as settings_row:
                        models_path = gr.Textbox(
                            max_lines=1,
                            label="Models Path",
                            value=models_path_default,
                        )

        load_btn.click(
            load_model,
            inputs=[state, model_selection],
            outputs=[state]
        ).then(
            state_change,
            inputs=[state],
            outputs=[state, load_btn, unload_btn]
        )

        unload_btn.click(
            unload_model,
            inputs=[state],
            outputs=[state]
        ).then(
            state_change,
            inputs=[state],
            outputs=[state, load_btn, unload_btn]
        )

        models_path.change(
            models_path_update,
            inputs=[state, models_path],
            outputs=[state, model_selection]
        )

        model_selection.change(
            model_selection_update,
            [state, model_selection],
            [state]
        )

        ui.load(
            load_ui,
            [state, models_path],
            [state, model_selection]
        )
    return ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--models-path", type=str, default="./")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    ui = build_ui(args.models_path)

    ui.launch(
        server_name=args.host,
        server_port=args.port,
        debug=True,
        max_threads=10
    )
