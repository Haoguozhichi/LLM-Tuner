import os

from openai import OpenAI
from transformers.utils.versions import require_version


require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


def main():
    client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
    )
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Output the color and number of each box."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/boxes.png"},
                },
            ],
        }
    )
    result = client.chat.completions.create(messages=messages, model="test")
    messages.append(result.choices[0].message)
    print("Round 1:", result.choices[0].message.content)
    # The image shows a pyramid of colored blocks with numbers on them. Here are the colors and numbers of ...
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What kind of flower is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/flowers.jpg"},
                },
            ],
        }
    )
    result = client.chat.completions.create(messages=messages, model="test")
    messages.append(result.choices[0].message)
    print("Round 2:", result.choices[0].message.content)
    # The image shows a cluster of forget-me-not flowers. Forget-me-nots are small ...


if __name__ == "__main__":
    main()
