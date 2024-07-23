import argparse
import json
from tqdm import tqdm
from glob import glob
from mix_eval.utils.common_utils import set_seed

import weave


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/mixeval-2024-06-01",
        required=True,
        help="Benchmark to evaluate."
    )

    return parser.parse_args()


def publish_data(args):
    """Function to log the mixeval jsonl data to W&B weave.
    """
    data_path = args.data_path
    files = glob(f"{data_path}/*/*.json")
    for file_path in tqdm(files):
        with open(file_path, 'r') as file:
            data = json.load(file)
        assert type(data) == dict
        print(len(data))
        data_rows = list(data.values())
        print(len(data_rows))
        assert type(data_rows) == list
        weave_dataset = weave.Dataset(
            name=f"{'_'.join(file_path.split('/')[-2:])}",
            rows=data_rows,
        )
        weave.publish(weave_dataset)


if __name__ == "__main__":
    weave.init("ayush-thakur/weave-mixeval")

    set_seed()
    args = parse_args()
    publish_data(args)
