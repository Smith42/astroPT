import datasets
from PIL import Image
import pandas as pd
import os
from pathlib import Path

with open("train.txt", "r") as fi:
    TRAIN = fi.read().splitlines()
with open("test.txt", "r") as fi:
    TEST = fi.read().splitlines()
with open("valid.txt", "r") as fi:
    VALID = fi.read().splitlines()

class MyDataset(datasets.GeneratorBasedBuilder):
    """My image dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "dr8_id": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        #metadata_path = "metadata.parquet"
        #metadata_df = pd.read_parquet(metadata_path)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "images": TRAIN,
                    #"metadata_df": metadata_df,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "images": TEST,
                    #"metadata_df": metadata_df,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "images": VALID,
                    #"metadata_df": metadata_df,
                },
            ),
        ]

    def _generate_examples(self, images, metadata_df=None):
        for image_path in images:
            dr8_id = Path(image_path).stem
            #metadata = metadata_df[metadata_df["dr8_id"] == dr8_id].to_dict("records")[0]
            yield image_path, {
                "image": image_path,
                "dr8_id": dr8_id,
            }

if __name__ == "__main__":
    dataset = datasets.load_dataset(
        "./galaxies.py",
        trust_remote_code=True,
        cache_dir="./cache/",
        ignore_verifications=True,
        num_proc=32,
    )
        
    # Upload the dataset to Huggingface Hub
    dataset.push_to_hub(
        REPO_LINK,
        max_shard_size="5GB",
    )
