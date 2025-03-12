from collections import defaultdict

import fire
from tqdm import tqdm

from llmtuner.data import get_dataset, get_template_and_fix_tokenizer
from llmtuner.hparams import get_train_args
from llmtuner.model import load_tokenizer


def length_cdf(
    model_name_or_path: str,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    interval: int = 1000,
):
    r"""Calculate the distribution of the input lengths in the dataset.

    Usage: export CUDA_VISIBLE_DEVICES=0
    python length_cdf.py --model_name_or_path path_to_model --dataset alpaca_en_demo --template default
    """
    model_args, data_args, training_args, _, _ = get_train_args(
        dict(
            stage="sft",
            model_name_or_path=model_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=1_000_000,
            preprocessing_num_workers=16,
            output_dir="dummy_dir",
            overwrite_cache=True,
            do_train=True,
        )
    )
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    trainset = get_dataset(template, model_args, data_args, training_args, "sft", **tokenizer_module)["train_dataset"]
    total_num = len(trainset)
    length_dict = defaultdict(int)
    for sample in tqdm(trainset["input_ids"], desc="Collecting lengths"):
        length_dict[len(sample) // interval * interval] += 1

    length_tuples = list(length_dict.items())
    length_tuples.sort()
    count_accu, prob_accu = 0, 0
    for length, count in length_tuples:
        count_accu += count
        prob_accu += count / total_num * 100
        print(f"{count_accu:d} ({prob_accu:.2f}%) samples have length < {length + interval}.")


if __name__ == "__main__":
    fire.Fire(length_cdf)
