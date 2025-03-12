import os
import re

from setuptools import find_packages, setup


def get_version() -> str:
    with open(os.path.join("src", "llmtuner", "extras", "env.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("VERSION")
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires() -> list[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


def get_console_scripts() -> list[str]:
    console_scripts = ["llmtuner-cli = llmtuner.cli:main"]
    if os.getenv("ENABLE_SHORT_CONSOLE", "1").lower() in ["true", "y", "1"]:
        console_scripts.append("lmt = llmtuner.cli:main")

    return console_scripts


extra_require = {
    "torch": ["torch>=1.13.1"],
    "torch-npu": ["torch==2.4.0", "torch-npu==2.4.0.post2", "decorator"],
    "metrics": ["nltk", "jieba", "rouge-chinese"],
    "deepspeed": ["deepspeed>=0.10.0,<=0.16.4"],
    "liger-kernel": ["liger-kernel"],
    "bitsandbytes": ["bitsandbytes>=0.39.0"],
    "hqq": ["hqq"],
    "eetq": ["eetq"],
    "gptq": ["optimum>=1.17.0", "auto-gptq>=0.5.0"],
    "awq": ["autoawq"],
    "aqlm": ["aqlm[gpu]>=1.1.0"],
    "vllm": ["vllm>=0.4.3,<=0.7.3"],
    "galore": ["galore-torch"],
    "apollo": ["apollo-torch"],
    "badam": ["badam>=1.2.1"],
    "adam-mini": ["adam-mini"],
    "qwen": ["transformers_stream_generator"],
    "minicpm_v": [
        "soundfile",
        "torchvision",
        "torchaudio",
        "vector_quantize_pytorch",
        "vocos",
        "msgpack",
        "referencing",
        "jsonschema_specifications",
    ],
    "modelscope": ["modelscope"],
    "openmind": ["openmind"],
    "swanlab": ["swanlab"],
    "dev": ["pre-commit", "ruff", "pytest"],
}


def main():
    setup(
        name="llmtuner",
        version=get_version(),
        author="steven",
        author_email="genglong@usc.edu",
        description="Fine-Tuning of 100+ LLMs",
        keywords=["AI", "LLM", "GPT", "ChatGPT", "Llama", "Transformer", "DeepSeek", "Pytorch"],
        url="https://github.com/Haoguozhichi/LLM-Tuner",
        package_dir={"": "src"},
        packages=find_packages("src"),
        python_requires=">=3.9.0",
        install_requires=get_requires(),
        extras_require=extra_require,
        entry_points={"console_scripts": get_console_scripts()},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main()
