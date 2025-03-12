import os
import random

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.train.test_utils import load_dataset_module


DEMO_DATA = os.getenv("DEMO_DATA", "llmtuner/demo_data")

TINY_LLAMA = os.getenv("TINY_LLAMA", "llmtuner/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "stage": "kto",
    "do_train": True,
    "finetuning_type": "full",
    "dataset": "kto_en_demo",
    "dataset_dir": "REMOTE:" + DEMO_DATA,
    "template": "llama3",
    "cutoff_len": 8192,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


@pytest.mark.parametrize("num_samples", [16])
def test_feedback_data(num_samples: int):
    train_dataset = load_dataset_module(**TRAIN_ARGS)["train_dataset"]
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    original_data = load_dataset(DEMO_DATA, name="kto_en_demo", split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        messages = original_data["messages"][index]
        ref_input_ids = ref_tokenizer.apply_chat_template(messages)
        prompt_len = len(ref_tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True))
        ref_labels = [IGNORE_INDEX] * prompt_len + ref_input_ids[prompt_len:]
        assert train_dataset["input_ids"][index] == ref_input_ids
        assert train_dataset["labels"][index] == ref_labels
        assert train_dataset["kto_tags"][index] == original_data["label"][index]
