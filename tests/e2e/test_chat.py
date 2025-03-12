import os

from llmtuner.chat import ChatModel


TINY_LLAMA = os.getenv("TINY_LLAMA", "llmtuner/tiny-random-Llama-3")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "finetuning_type": "lora",
    "template": "llama3",
    "infer_dtype": "float16",
    "do_sample": False,
    "max_new_tokens": 1,
}

MESSAGES = [
    {"role": "user", "content": "Hi"},
]

EXPECTED_RESPONSE = "_rho"


def test_chat():
    chat_model = ChatModel(INFER_ARGS)
    assert chat_model.chat(MESSAGES)[0].response_text == EXPECTED_RESPONSE


def test_stream_chat():
    chat_model = ChatModel(INFER_ARGS)
    response = ""
    for token in chat_model.stream_chat(MESSAGES):
        response += token

    assert response == EXPECTED_RESPONSE
