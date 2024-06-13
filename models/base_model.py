import warnings

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')
        torch.set_default_device("cuda")
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

    def load(self, model, device=0):
        if model is not None:
            model.unload()
        self._load(device)

    def _load(self, device=0):
        self.unload()
        torch.cuda.set_device(int(device))
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": int(device)},
            trust_remote_code=True
        ).to(torch.bfloat16).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        return self.model, self.tokenizer

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
