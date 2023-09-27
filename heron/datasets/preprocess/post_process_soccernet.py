from pathlib import Path
import pandas as pd
import ast

SPLIT = "valid"

DATA_DIR = Path("data/Soccernet")

INVALID_CSV_PATH = DATA_DIR / f"invalid_data_{SPLIT}.csv"
CAPTION_CSV_PATH = DATA_DIR / f"soccernet_{SPLIT}.csv"
METADETA_CSV_PATH = DATA_DIR / f"soccernet_{SPLIT}_game_subinfo.csv"

MERGED_INVALID_CSV_PATH = DATA_DIR / f"soccernet_{SPLIT}_merged.csv"
MERGED_CLEANED_CSV_PATH = DATA_DIR / f"soccernet_{SPLIT}_merged_cleaned.csv"


def check():
    df = pd.read_csv(MERGED_CLEANED_CSV_PATH, delimiter=",")
    img_list = ast.literal_eval(df.loc[0, "img_path"])
    print("%10d" % df.loc[0, "videoID"])
    print(img_list[0])


invalid_df = pd.read_csv(INVALID_CSV_PATH, delimiter=",", dtype=str)
caption_df = pd.read_csv(CAPTION_CSV_PATH, delimiter=",", dtype=str)
metadata_df = pd.read_csv(METADETA_CSV_PATH, delimiter=",", dtype=str)


def save_modified_invalid_csv():
    # Extract the portion after "raw_images/" as videoID
    invalid_df["videoID"] = invalid_df["dst_images_path"].str.extract(
        r"raw_images/(.*)"
    )

    # Reorder the columns to place videoID as the first column again
    invalid_df = invalid_df[
        ["videoID", "dst_images_path", "src_video_path", "spotTime", "message"]
    ]

    invalid_df.to_csv(INVALID_CSV_PATH, index=False)


save_modified_invalid_csv()

merged_train_df = pd.merge(
    caption_df, metadata_df, how="left", on="caption"
).drop_duplicates(subset=["caption"])


def save_invalid_csv():
    merged_invalid_train_df = pd.merge(
        merged_train_df, invalid_df, how="left", on="videoID"
    ).drop_duplicates(subset=["videoID"])

    merged_invalid_train_df.to_csv(MERGED_INVALID_CSV_PATH, index=False)


save_invalid_csv()


def save_cleaned_csv():
    # Get the list of invalid videoIDs
    invalid_video_ids = invalid_df["videoID"].tolist()

    # Filter out rows with invalid videoIDs from merged_train_df
    merged_cleaned_df = merged_train_df.loc[
        ~merged_train_df["videoID"].isin(invalid_video_ids)
    ]

    # Save the cleaned DataFrame to CSV
    merged_cleaned_df.to_csv(MERGED_CLEANED_CSV_PATH, index=False)


save_cleaned_csv()
