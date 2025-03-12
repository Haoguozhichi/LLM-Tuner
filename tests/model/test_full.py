import os

import torch

from llmtuner.train.test_utils import load_infer_model, load_train_model


TINY_LLAMA = os.getenv("TINY_LLAMA", "llmtuner/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "full",
    "dataset": "llmtuner/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "finetuning_type": "full",
    "template": "llama3",
    "infer_dtype": "float16",
}


def test_full_train():
    model = load_train_model(**TRAIN_ARGS)
    for param in model.parameters():
        assert param.requires_grad is True
        assert param.dtype == torch.float32


def test_full_inference():
    model = load_infer_model(**INFER_ARGS)
    for param in model.parameters():
        assert param.requires_grad is False
        assert param.dtype == torch.float16
