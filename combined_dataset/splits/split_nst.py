import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import re
from sklearn.model_selection import train_test_split

# Parser
parser = argparse.ArgumentParser(
    description="Split dataset in train/test/eval close to a 80/10/10 proportion"
)

parser.add_argument(
    "-d", "--csv_dir", type=str, required=True, help="Path to csv with NST data",
)

# Cleaning functions
lownums = [
    "én",
    "ett",
    "en",
    "to",
    "tre",
    "fire",
    "fem",
    "seks",
    "syv",
    "sju",
    "åtte",
    "ni",
]
ten = ["ti"]
teen = [
    "elleve",
    "tolv",
    "tretten",
    "fjorten",
    "femten",
    "seksten",
    "sytten",
    "atten",
    "nitten",
]
tens = [
    "tyve",
    "tjue",
    "tredve",
    "tretti",
    "førti",
    "femti",
    "seksti",
    "sytti",
    "åtti",
    "nitti",
]
large = ["hundre", "tusen", "million", "milliard", "millioner", "milliarder"]
composed = []
for t in tens:
    for l in lownums:
        composed.append(t + l)
        composed.append(l + "og" + t)
nums = lownums + ten + teen + tens + large + composed + ["og"]

namepattern = re.compile(r"^[A-ZÆØÅ][a-zæøå]+$")


def _is_all_nums(listlike):
    returnval = True
    for l in listlike:
        if l not in nums:
            returnval = False
            break
    return returnval


def _check_name(listlike):
    returnval = True
    for l in listlike:
        if not namepattern.match(l):
            returnval = False
            break
    return returnval


def _check_spell(listlike):
    returnval = True
    for l in listlike:
        if not len(l) == 1:
            returnval = False
            break
    return returnval


def _check_repeated_words(listlike):
    returnval = True
    firstword = listlike[0]
    for l in listlike:
        if l != firstword:
            returnval = False
    return returnval


def is_clean(unstandardized, standardized):
    clean = True
    if _is_all_nums(standardized):
        clean = False
    elif (
        "punktum" in standardized
        or "komma" in standardized
        or "(...Vær" in unstandardized
    ):
        clean = False
    elif _check_name(unstandardized):
        clean = False
    elif _check_spell(standardized):
        clean = False
    elif _check_repeated_words(standardized):
        clean = False
    return clean


logger = logging.getLogger(__name__)

# Auxiliary function
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
    df = pd.read_csv(csv_dir, names=cols)
    print("Dropping segments without transcription")
    df.dropna(subset="standardized_text", inplace=True)
    print("Dropping segments in Nynorsk")
    df = df[df.language == "nb-NO"]
    print(
        "Dropping segments containing names, numbers, spellings etc., and segments with read punctuation"
    )
    df = df[
        df.apply(
            lambda row: is_clean(
                row.raw_text.split(" "), row.standardized_text.split(" ")
            ),
            axis=1,
        )
    ]
    df_train = df[df.original_data_split == "nst_train"].copy()
    df_orig_test = df[df.original_data_split == "nst_test"].copy()
    df_test, df_eval = train_test_split(df_orig_test, test_size=0.5, random_state=0)

    for df, name, i in zip(
        [df_train, df_eval, df_test], ["Train", "Eval", "Test"], [0, 1, 2]
    ):
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
