from SoccerNet.Downloader import SoccerNetDownloader as SNdl
from SoccerNet.utils import getListGames
from pathlib import Path

PASSWORD = ""

SOCCERNET_PATH = "/raid_elmo/home/lr/moriy/SoccerNet"

if __name__ == "__main__":
    mySNdl = SNdl(LocalDirectory="/Users/heste/Downloads/SoccerNet")
    mySNdl.password = PASSWORD
    mySNdl.downloadGames(
        files=["1_720p.mkv", "2_720p.mkv", "Labels-caption.json", "Labels.json"],
        split=["train", "valid", "test", "challenge"],
        task="caption",
        verbose=1,
    )
