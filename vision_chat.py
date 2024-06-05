import argparse
import logging as logger
import gradio as gr
import os

from models.bunny import BunnyModel
from models.cogvlm2 import CogVLM2Model


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
