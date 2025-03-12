import os

import pytest

from llmtuner.train.test_utils import compare_model, load_infer_model, load_reference_model, patch_valuehead_model


TINY_LLAMA = os.getenv("TINY_LLAMA", "llmtuner/tiny-random-Llama-3")

TINY_LLAMA_VALUEHEAD = os.getenv("TINY_LLAMA_VALUEHEAD", "llmtuner/tiny-random-Llama-3-valuehead")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "template": "llama3",
    "infer_dtype": "float16",
}


@pytest.fixture
def fix_valuehead_cpu_loading():
    patch_valuehead_model()


def test_base():
    model = load_infer_model(**INFER_ARGS)
    ref_model = load_reference_model(TINY_LLAMA)
    compare_model(model, ref_model)


@pytest.mark.usefixtures("fix_valuehead_cpu_loading")
def test_valuehead():
    model = load_infer_model(add_valuehead=True, **INFER_ARGS)
    ref_model = load_reference_model(TINY_LLAMA_VALUEHEAD, add_valuehead=True)
    compare_model(model, ref_model)
