# Copyright 2023 Yuchiro Mori. All rights reserved.
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

from __future__ import annotations
from SoccerNet.Downloader import getListGames
from SoccerNet.Evaluation.utils import getMetaDataTask
import json
import os
import pandas as pd
from tqdm import tqdm
import random
import string
from dataclasses import dataclass
import cv2
import sys
import logging
from datetime import datetime
from argparse import ArgumentParser
import csv
import multiprocessing
from multiprocessing import Pool
import subprocess


SOCCERNET_PATH = "/raid_elmo/home/lr/moriy/SoccerNet"   # noqa


DST_DATASET_PATH = "/raid_elmo/home/lr/moriy/heron/data/SoccerNet"  # noqa

IMAGE_FILENAME_FORMAT = "frame_%04d.jpg"


_VIDDEO_BASENAME = (
    "1_224p.mkv",
    "2_224p.mkv",
)


RANDOM_NAMES = set()


# TODO: videoIDではなくて、captionIDじゃないか？
@dataclass(frozen=True)
class SingleData:
    caption: str
    videoPath: str
    videoID: str
    spotTime: int | float


def get_max_len_seconds(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)  # 動画を読み込む
    video_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # フレーム数を取得する
    video_fps = cap.get(cv2.CAP_PROP_FPS)  # フレームレートを取得する
    video_len_sec = video_frame_count / video_fps  # 長さ（秒）を計算する

    return video_len_sec


def random_name(n):
    randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
    name = "".join(randlst)
    if name in RANDOM_NAMES:
        return random_name(n)

    RANDOM_NAMES.add(name)

    return name


def extract_timeinfo(time_string: str):
    half = int(time_string[0])

    minutes, seconds = time_string.split(" ")[-1].split(":")
    minutes, seconds = int(minutes), int(seconds)

    return half, minutes, seconds


class Stage1:
    def __init__(self, path, split) -> None:
        self.path = path
        self.split = split
        self.listGames = getListGames([split], task="caption")
        labels, num_classes, dict_event, _ = getMetaDataTask("caption", "SoccerNet", 2)
        self.labels = labels
        self.num_classes = num_classes
        self.dict_event = dict_event

    def run(self):
        res = []
        total = 0
        for game_i, game in tqdm(enumerate(self.listGames)):
            label_path = os.path.join(self.path, game, self.labels)
            game_path = os.path.join(self.path, game)
            labels = json.load(open(label_path))

            for annotation in labels["annotations"]:
                time = annotation["gameTime"]
                half, minutes, seconds = extract_timeinfo(time)

                event = annotation["label"]
                if event not in self.dict_event or half > 2:
                    continue

                caption = annotation["anonymized"]
                videoPath = os.path.join(game_path, _VIDDEO_BASENAME[half - 1])
                videoID = "%010d" % total
                spotTime = minutes * 60 + seconds
                res.append(SingleData(caption, videoPath, videoID, spotTime))
                # index += 1
                total += 1

        return res


class Stage2:
    """ """

    def __init__(self, dst_dir, split, window_size, framerate) -> None:
        self.window_size = window_size
        self.framerate = framerate
        self.dst_dir = dst_dir
        self.split = split

        self.invalid_data_file = os.path.join(self.dst_dir, f"invalid_data_{self.split}.csv") # noqa
        self.csv_path = os.path.join(self.dst_dir, f"soccernet_{self.split}.csv") # noqa
        self.csv_game_subinfo_path = os.path.join(self.dst_dir, f"soccernet_{self.split}_game_subinfo.csv") # noqa

        os.makedirs(self.dst_dir, exist_ok=True)

    def run(self, datas: list[SingleData]):
        if datas is None:
            raise ValueError("datas is None")

        prompt_list = []
        img_path_origin_list = []
        caption_list = []
        extract_video_args = []

        # imageの保存先
        dst_image_dir = os.path.join(self.dst_dir, "raw_images")

        for data in datas:
            prompt = ""
            caption = data.caption
            videoID = data.videoID
            src_video_path = data.videoPath
            spotTime = data.spotTime

            dst_images_path = os.path.join(dst_image_dir, videoID)
            img_path_origin_list.append(dst_images_path)
            prompt_list.append(prompt)
            caption_list.append(caption)

            extract_video_args.append((src_video_path, dst_images_path, spotTime))

        # この記述がないと関数をpickel化できない
        global _video_clipping_worker

        def _video_clipping_worker(args):
            status, message = self.extract_video_to_images(*args)
            if not status:
                logger.warning(f"Failed to extract video: {repr(args)}")
                logger.warning(message)
                # args = (src_video_path, dst_video_path, spotTime)
                # return dst_video_path, src_video_path, spotTime, message
                return args[1], args[0], args[2], message
            return None

        with Pool(multiprocessing.cpu_count()) as p:
            bad_results = list(
                tqdm(
                    p.imap(_video_clipping_worker, extract_video_args),
                    total=len(extract_video_args),
                )
            )
        self.write_invalid_data(bad_results)

        img_path_list = []
        for img_path_origin in img_path_origin_list:
            sorted_img_path_list = sorted(os.listdir(img_path_origin))
            img_path_list.append(
                sorted_img_path_list
            )

        # write csv
        df = pd.DataFrame(
            {
                "img_path": img_path_list,
                "prompt": prompt_list,
                "caption": caption_list,
            }
        )
        df.to_csv(self.csv_path, index=False)

        # write game subinfo csv (1 on 1 correspondence with csv of written images)
        df = pd.DataFrame(
            {
                "videoID": [data.videoID for data in datas],
                "videoPath": [data.videoPath for data in datas],
                "spotTime": [data.spotTime for data in datas],
                "caption": [data.caption for data in datas],
            }
        )
        df.to_csv(self.csv_game_subinfo_path, index=False)

    def write_invalid_data(self, results):
        invalid_data = [res for res in results if res is not None]
        invalid_data = pd.DataFrame(
            invalid_data,
            columns=["dst_video_path", "src_video_path", "spotTime", "message"]
        )
        invalid_data.to_csv(self.invalid_data_file, index=False) # noqa

    def extract_video_to_images(
        self, src_video_path: str, dst_image_dir: str, spotTime: int | float
    ):
        # TODO: 並列処理の効率化のために、一連のエラーハンドリングは上位で行いたい
        # Ensure the destination directory exists or create it
        if not os.path.exists(dst_image_dir):
            os.makedirs(dst_image_dir)

        if not os.path.exists(src_video_path):
            return False, f"Video file does not exist: {src_video_path}"

        maxLenSeconds = get_max_len_seconds(src_video_path)

        start_time = max(0, spotTime - self.window_size / 2)
        end_time = min(maxLenSeconds, spotTime + self.window_size / 2)
        duration = max(0, min(self.window_size, end_time - start_time))
        framerate = self.framerate
        if end_time - start_time <= 0:
            return (
                False,
                f"Invalid time range, duration:{duration}, start_time:{start_time}, end_time:{end_time}",
            )

        # Format for saving images: "frame_%04d.jpg"
        # This will save images as frame_0001.jpg, frame_0002.jpg, etc.
        dst_image_format = os.path.join(dst_image_dir, IMAGE_FILENAME_FORMAT)

        command = [
            "ffmpeg",
            "-i",
            '"%s"' % src_video_path,
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-vf",
            f"fps={framerate}",  # Extract images at the desired frame rate
            '"%s"' % dst_image_format,
            "-threads",
            "1",
            "-loglevel",
            "panic",
        ]
        command = " ".join(command)
        try:
            _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            return False, repr(err.output)

        # Count the number of images saved
        num_images_saved = len(os.listdir(dst_image_dir))

        if num_images_saved > 0:
            return True, f"{num_images_saved} images saved."
        else:
            return False, "Failed to save images."


def main(args):
    # Set path to SoccerNet data
    # Images are download from here: https://www.soccer-net.org/data

    stage1 = Stage1(args.soccernet_path, args.split)
    datas = stage1.run()

    stage2 = Stage2(
        args.dst_data_path, args.split, args.window_size, args.framerate
    )
    stage2.run(datas)


if __name__ == "__main__":
    # argparse
    parser = ArgumentParser()
    parser.add_argument("--soccernet-path", type=str, default=SOCCERNET_PATH)
    parser.add_argument("--dst-data-path", type=str, default=DST_DATASET_PATH)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--framerate", type=int, default=1)

    args = parser.parse_args()

    # logging
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(filename=f"{now_str}_{args.split}_result.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)

    # main
    main(args)
