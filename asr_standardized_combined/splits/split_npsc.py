import logging
import argparse
import pandas as pd
from pathlib import Path

# Parser
parser = argparse.ArgumentParser(
    description="Split dataset in train/test/eval, following the original splits in the NPSC"
)

parser.add_argument(
    "-d", "--csv_dir", type=str, required=True, help="Path to csv with NPSC data",
)

logger = logging.getLogger(__name__)


def speaker_overlap(list1, list2):
    c = 0
    for e1 in list1:
        if e1 in list2:
            c += 1
    return c


cols = [
    "speaker_id",
    "gender",
    "utterance_id",
    "language",
    "raw_text",
    "full_audio_file",
    "original_data_split",
    "region",
    "duration",
    "start",
    "end",
    "utterance_audio_file",
    "standardized_text",
]


def split_data(csv_dir):
    total_df = pd.read_csv(csv_dir, names=cols)
    df_train = total_df[total_df.original_data_split == "npsc_train"].copy()
    df_test = total_df[total_df.original_data_split == "npsc_test"].copy()
    df_eval = total_df[total_df.original_data_split == "npsc_eval"].copy()

    for df, name in zip([df_train, df_eval, df_test], ["Train", "Eval", "Test"]):
        print("*** {} ***".format(name))
        print(df.shape[0], "utterances")
        total_time = df.duration.sum()
        print("Duration in hours: {}".format(round(1 / 3600 * total_time, 2)))
        list_speakers = df.speaker_id.unique()
        print("Number of different speakers: {}".format(len(list_speakers)))
        print("Gender (in terms of number of utterances):")
        for line in df.gender.value_counts(dropna=False, normalize=True).items():
            print(" {}: {}%".format(line[0], round(100 * line[1], 2)))
        print("Gender (in terms of time speaking):")
        for line in df.groupby("gender").duration.sum().items():
            print(" {}: {}%".format(line[0], round(100 * line[1] / total_time, 2)))
        print("Dialect group:")
        for line in df.region.value_counts(dropna=False, normalize=True).items():
            print(" {}: {}%".format(line[0], round(100 * line[1], 2)))
        # Creating split dir and saving csv
        csv_path = Path(csv_dir)
        split_dir = csv_path.parent / name
        print("Creating split directory {}".format(split_dir))
        split_dir.mkdir(exist_ok=True)
        head_filename = csv_path.stem
        print(
            "Saving csv {}_{}.csv in {}".format(head_filename, name.lower(), split_dir)
        )
        df.to_csv(
            "{}/{}_{}.csv".format(split_dir, head_filename, name.lower()),
            header=False,
            index=False,
        )
        print()

    speakers_train = df_train.speaker_id.unique()
    speakers_eval = df_eval.speaker_id.unique()
    speakers_test = df_test.speaker_id.unique()

    print("*** Overlaps in speakers ***")
    print(
        "Train-Eval: {} speakers in common".format(
            speaker_overlap(speakers_train, speakers_eval)
        )
    )
    print(
        "Train-Test: {} speakers in common".format(
            speaker_overlap(speakers_train, speakers_test)
        )
    )
    print(
        "Test-Eval: {} speakers in common".format(
            speaker_overlap(speakers_test, speakers_eval)
        )
    )


if __name__ == "__main__":

    args = parser.parse_args()

    # Options chosen
    logger.info("Splitting data from {}".format(args.csv_dir))

    split_data(args.csv_dir)
