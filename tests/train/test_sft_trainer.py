import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from transformers import DataCollatorWithPadding

from llmtuner.data import get_dataset, get_template_and_fix_tokenizer
from llmtuner.hparams import get_train_args
from llmtuner.model import load_model, load_tokenizer
from llmtuner.train.sft.trainer import CustomSeq2SeqTrainer


DEMO_DATA = os.getenv("DEMO_DATA", "llmtuner/demo_data")

TINY_LLAMA = os.getenv("TINY_LLAMA", "llmtuner/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "dataset": "llmtuner/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "max_steps": 1,
}


@dataclass
class DataCollatorWithVerbose(DataCollatorWithPadding):
    verbose_list: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        self.verbose_list.extend(features)
        batch = super().__call__(features)
        return {k: v[:, :1] for k, v in batch.items()}  # truncate input length


@pytest.mark.parametrize("disable_shuffling", [False, True])
def test_shuffle(disable_shuffling: bool):
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        {
            "output_dir": os.path.join("output", f"shuffle{str(disable_shuffling).lower()}"),
            "disable_shuffling": disable_shuffling,
            **TRAIN_ARGS,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = DataCollatorWithVerbose(tokenizer=tokenizer)
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        **dataset_module,
        **tokenizer_module,
    )
    trainer.train()
    if disable_shuffling:
        assert data_collator.verbose_list[0]["input_ids"] == dataset_module["train_dataset"][0]["input_ids"]
    else:
        assert data_collator.verbose_list[0]["input_ids"] != dataset_module["train_dataset"][0]["input_ids"]
