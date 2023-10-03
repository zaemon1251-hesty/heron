# Copyright 2023 Turing Inc. Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import cv2
import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset
import pandas as pd
import os
import ast
from .base_datasets import ResilientDataset

HFProcessor = "HFProcessor"


class SoccerNetDataset(ResilientDataset):
    """Dataset for M3IT Dataset learning"""

    def __init__(
        self,
        loaded_dataset: ConcatDataset,
        processor: HFProcessor,
        max_length: int,
        is_inference: bool = False,
        dataset_root: str = "./",
        num_frames: int = 16,
    ):
        super(SoccerNetDataset, self).__init__(is_inference)
        self.loaded_dataset = loaded_dataset
        self.max_length = max_length
        self.processor = processor
        self.is_inference = is_inference
        self.dataset_root = dataset_root
        self.num_frames = num_frames

    @classmethod
    def create(
        cls,
        dataset_config: dict,
        processor: HFProcessor,
        max_length: int,
        split: str = "train",
        is_inference: bool = False,
    ):
        dataset_root = dataset_config["dataset_root"]
        num_frames = dataset_config["num_frames"]
        target_dataset = pd.DataFrame()

        if split == "train":
            df_train = pd.read_csv(
                os.path.join(
                    dataset_root, "data/SoccerNet/soccernet_train_merged_cleaned.csv"
                ),
                delimiter=",",
            )
            target_dataset = df_train.sample(n=250, random_state=42)
        else:
            df_valid = pd.read_csv(
                os.path.join(
                    dataset_root, "data/SoccerNet/soccernet_valid_merged_cleaned.csv"
                ),
                delimiter=",",
            )
            target_dataset = df_valid.sample(n=250, random_state=42)

        return cls(
            target_dataset,
            processor,
            max_length,
            is_inference=is_inference,
            dataset_root=dataset_root,
            num_frames=num_frames,
        )

    def __len__(self) -> int:
        return len(self.loaded_dataset)

    def _get_item_train(self, index):
        # cf: https://huggingface.co/datasets/MMInstruction/M3IT#data-instances
        row = self.loaded_dataset.iloc[index]

        # some of nlvr data were broken
        caption = row["caption"]  # str
        prompt = row["prompt"]  # str
        text = f"##Instrcution:{prompt} \n##Caption:{caption}"
        imgs = self._get_iamges(row["videoID"], row["img_path"])

        inputs = self.processor(
            images=imgs,
            text=text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        # batch size 1 -> unbatch
        inputs = {k: v[0] for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"]
        return inputs

    def _get_item_inference(self, index):
        # cf: https://huggingface.co/datasets/MMInstruction/M3IT#data-instances
        row = self.loaded_dataset.iloc[index]
        # some of nlvr data were broken
        prompt = row["prompt"]  # str
        answer = row["caption"]  # str
        text = f"##Instrcution:{prompt} \n##Caption:"

        imgs = self._get_iamges(row["videoID"], row["img_path"])

        inputs = self.processor(
            images=imgs,
            text=text,
            return_tensors="pt",
        )
        inputs["labels"] = None
        return inputs, answer

    def _get_iamges(self, video_id, image_paths_str):
        video_id = "%10d" % video_id
        # imageのロード
        img_path_list: list
        img_path_list = ast.literal_eval(image_paths_str)

        # 画像の枚数が16枚になるようにする
        while len(img_path_list) < self.num_frames:
            img_path_list.append(img_path_list[-1])
            if len(img_path_list) == self.num_frames:
                break
            img_path_list.insert(0, img_path_list[0])

        imgs = []
        for img_path in img_path_list:
            # TODO 色々歪な処理なので直す
            # TODO raw_images → raw_images_{split}に変更する
            exact_img_path = os.path.join(
                self.dataset_root, "data/SoccerNet/raw_images", video_id, img_path
            )
            img = Image.open(exact_img_path).convert("RGB")
            img = np.array(img)
            if img.shape[2] != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imgs.append(img)
        return imgs


if __name__ == "__main__":
    dataset_config = {}
    processor = None
    max_length = 0
    train_datsaet = SoccerNetDataset.create(
        dataset_config, processor, max_length, "train"
    )
    print(train_datsaet[0])
