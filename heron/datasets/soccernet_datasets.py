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


from base64 import b64decode
from io import BytesIO

import cv2
import datasets
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
    ):
        super(SoccerNetDataset, self).__init__(is_inference)
        self.loaded_dataset = loaded_dataset
        self.max_length = max_length
        self.processor = processor
        self.is_inference = is_inference
        self.dataset_root = dataset_root

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
        target_dataset = pd.DataFrame()

        if split == "train":
            df_train = pd.read_csv(
                os.path.join(
                    dataset_root, "data/SoccerNet/soccernet_train_merged_cleaned.csv"
                )
            )
            target_dataset = df_train
        else:
            # TODO validationのデータセットを作る
            pass

        return cls(
            target_dataset,
            processor,
            max_length,
            is_inference=is_inference,
            dataset_root=dataset_root,
        )

    def __len__(self) -> int:
        return len(self.loaded_dataset)

    def _get_item_train(self, index):
        # cf: https://huggingface.co/datasets/MMInstruction/M3IT#data-instances
        row = self.loaded_dataset.iloc[index]

        # some of nlvr data were broken
        caption = row["caption"]  # str

        # imageのロード
        img_path_list = ast.literal_eval(row["img_path"])
        imgs = []
        for img_path in img_path_list:
            video_id = "%10d" % row["videoID"]

            # TODO 色々歪な処理なので直す
            exact_img_path = os.path.join(
                self.dataset_root, "data/SoccerNet/raw_images", video_id, img_path
            )

            img = Image.open(exact_img_path).convert("RGB")
            img = np.array(img)
            if img.shape[2] != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imgs.append(img)

        inputs = self.processor(
            images=imgs,
            text=caption,
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
        # TODO  inferenceの実装
        return None, None, None

        # cf: https://huggingface.co/datasets/MMInstruction/M3IT#data-instances
        # row = self.loaded_dataset[index]

        # # some of nlvr data were broken
        # instruction = row["instruction"]  # str
        # question = row["inputs"]  # str
        # answer = row["outputs"]  # str
        # text = f"##Instruction: {instruction} ##Question: {question} ##Answer: "

        # # imageのロード
        # image_base64_str_list = row["image_base64_str"]  # str (base64)
        # img = Image.open(BytesIO(b64decode(image_base64_str_list[0]))).convert("RGB")
        # img = np.array(img)
        # if img.shape[2] != 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # inputs = self.processor(
        #     text,
        #     img,
        #     return_tensors="pt",
        # )
        # inputs["labels"] = None
        # return inputs, img, answer


if __name__ == "__main__":
    dataset_config = {}
    processor = None
    max_length = 0
    train_datsaet = SoccerNetDataset.create(
        dataset_config, processor, max_length, "train"
    )
    print(train_datsaet[0])
