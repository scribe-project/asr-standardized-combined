import pandas as pd
import csv
import re
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from .shared_classes import consolidated_utterance, consolidated_utterance_phon
from pympi.Praat import TextGrid
import warnings


# Regex patterns

bg_level_pattern = re.compile(r"level=([a-z]+)")
bg_starttime_pattern = re.compile(r"time=(\d+\.?\d*)")
tn_speaker_pattern = re.compile(r"speaker=([a-z0-9]+)")
tn_mode_pattern = re.compile(r"mode=([a-z]+)")
tn_fidelity_pattern = re.compile(r"fidelity=([a-z]+)")
tn_channel_pattern = re.compile(r"channel=([a-z]+)")
sg_topic_pattern = re.compile(r"topic=([a-z0-9]+)")
type_pattern = re.compile(r"type=([a-z]+)")
start_pattern = re.compile(r"startTime=(\d+\.?\d*)")
end_pattern = re.compile(r"endTime=(\d+\.?\d*)")


# Regex pattern groupings

background_patterns = {
    "background_start": bg_starttime_pattern,
    "background_level": bg_level_pattern,
    "background_type": type_pattern,
}
turn_patterns = {
    "turn_speaker": tn_speaker_pattern,
    "turn_mode": tn_mode_pattern,
    "turn_start": start_pattern,
    "turn_end": end_pattern,
    "turn_fidelity": tn_fidelity_pattern,
    "turn_channel": tn_channel_pattern,
}
segment_patterns = {
    "segment_topic": sg_topic_pattern,
    "segment_start": start_pattern,
    "segment_end": end_pattern,
    "segment_type": type_pattern,
}

# Speaker dict rundkast phon
phon_speakers = {
    "TOK": ("male", "south"),
    "WEE": ("female", "east"),
    "GRA": ("female", "east"),
    "ARS": ("male", "east"),
    "BJH": ("male", "north"),
    "AET": ("male", "mid"),
    "ELS": ("female", "east"),
    "LIF": ("female", "east"),
    "SAP": ("female", "east"),
    "JEK": ("male", "east"),
}

# Functions


def make_linedict(line):
    """Parse a line of a .lab file and return its content as a dict with
    similar keys as in the .lab file"""
    mydict = {"start": line[0]}
    if len(line) == 5:
        mydict["transcription"] = line[1]
        for k, pattern in background_patterns.items():
            if pattern.search(line[2]):
                mydict[k] = pattern.search(line[2]).group(1)
            else:
                mydict[k] = np.nan
        for k, pattern in turn_patterns.items():
            if pattern.search(line[3]):
                mydict[k] = pattern.search(line[3]).group(1)
            else:
                mydict[k] = np.nan
        for k, pattern in segment_patterns.items():
            if pattern.search(line[4]):
                mydict[k] = pattern.search(line[4]).group(1)
            else:
                mydict[k] = np.nan
    elif len(line) > 2:
        mydict["transcription"] = line[1]
        for d in [background_patterns, turn_patterns, segment_patterns]:
            for k in d.keys():
                mydict[k] = np.nan
    else:
        mydict["transcription"] = np.nan
        for d in [background_patterns, turn_patterns, segment_patterns]:
            for k in d.keys():
                mydict[k] = np.nan
    return mydict


def get_language(accent):
    if re.match(".*bokmål.*", accent):
        return "nb-NO"
    elif re.match(".*nynorsk.*", accent):
        return "nn-NO"
    else:
        return "other"


def normalize_topics(topic):
    if isinstance(topic, str):
        topic = topic.lower()
    if topic == "annen":
        topic = "annet"
    return topic


def clean_dialect(dialectstring):
    """Clean up known errors in the Rundkast dialect annotations"""
    east_pattern = re.compile(r"østland|østlandet|østlanf|østlandsk|østnorsk")
    west_pattern = re.compile(r"vestland|vesltand")
    mid_pattern = re.compile(r"trøndelag|trønderlag|trøderlag|midtnorge|trønder")
    north_pattern = re.compile(r"nord-norge|nordland|nordnorge|nordnorsk|norland")
    south_pattern = re.compile(r"sørland|sørlandet|sørlandsk")
    dialect = "unknown"
    if isinstance(dialectstring, str):
        if east_pattern.search(dialectstring.lower()):
            dialect = "east"
        elif west_pattern.search(dialectstring.lower()):
            dialect = "west"
        elif mid_pattern.search(dialectstring.lower()):
            dialect = "mid"
        elif north_pattern.search(dialectstring.lower()):
            dialect = "north"
        elif south_pattern.search(dialectstring.lower()):
            dialect = "south"
    return dialect


def parse_corpus_files(rundkastdir):
    """Specify the root directory of Rundkast and produce a DataFrame of the Rundkast data"""

    transcriptiondir = Path(rundkastdir) / "transcription"

    trsdir = transcriptiondir / "trs"
    labdir = transcriptiondir / "lab"

    # BeatifulSoup produces warnings. Probably a bug
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    allstems = [x.stem for x in Path(trsdir).glob("*.trs")]
    all_datasets = []

    # Loop through the .lab and .trs files
    for stem in allstems:
        trsfile = Path(trsdir) / (stem + ".trs")
        labfile = Path(labdir) / (stem + ".lab")
        audiofile = Path(stem + ".wav")
        lines = []
        with labfile.open(mode="r", newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter="\t")
            for r in reader:
                lines.append(r)
        linedicts = [make_linedict(l) for l in lines]
        df = pd.DataFrame(linedicts)
        df.loc[:, "end"] = df.start.shift(-1)
        df.loc[:, "transcription_file"] = labfile.name
        df.loc[:, "full_audio_file"] = audiofile.name
        df.drop(df.tail(1).index, inplace=True)  # last line contains endtime only
        df = df.astype(
            {
                "start": float,
                "background_start": float,
                "turn_start": float,
                "turn_end": float,
                "segment_start": float,
                "segment_end": float,
                "end": float,
            }
        )
        with trsfile.open(mode="r") as trs_file:
            trs = trs_file.read()
        soup = BeautifulSoup(trs, "lxml")
        topics = []
        for t in soup.topics:
            if t.name == "topic":
                topics.append(t.attrs)
        speakers = []
        for s in soup.speakers:
            if s.name == "speaker":
                speakers.append(s.attrs)
        for s in speakers:
            s["language"] = get_language(s["accent"])
        topics_df = pd.DataFrame(topics)
        topics_df.loc[:, "desc"] = topics_df.desc.apply(lambda x: normalize_topics(x))
        speakers_df = pd.DataFrame(speakers)
        speakers_df.loc[:, "accent"] = speakers_df.accent.apply(
            lambda x: clean_dialect(x)
        )
        df = df.merge(speakers_df, left_on="turn_speaker", right_on="id", how="left")
        df = df.merge(topics_df, left_on="segment_topic", right_on="id", how="left")
        all_datasets.append(df)

    full_corpus_df = pd.concat(all_datasets)

    # Clean the data
    full_corpus_df.loc[:, "segment_topic"] = full_corpus_df.loc[:, "desc"]
    full_corpus_df = full_corpus_df[~full_corpus_df.transcription.isna()]
    full_corpus_df = full_corpus_df.drop(["id_x", "id_y", "desc"], axis=1)
    full_corpus_df.loc[:, "duration"] = full_corpus_df.end - full_corpus_df.start

    # Make speaker information consistent across files
    speakers = (
        full_corpus_df[
            ["name", "check", "type", "dialect", "accent", "scope", "language"]
        ]
        .dropna()
        .drop_duplicates()
    )
    speakers.reset_index(drop=True, inplace=True)
    speakers.loc[:, "identifier"] = speakers.index
    speakers.loc[:, "speaker_id"] = speakers.loc[:, "identifier"].apply(
        lambda x: f"speaker_{x}"
    )
    full_corpus_df = pd.merge(
        full_corpus_df,
        speakers[
            [
                "name",
                "check",
                "type",
                "dialect",
                "accent",
                "scope",
                "language",
                "speaker_id",
            ]
        ],
        on=["name", "check", "type", "dialect", "accent", "scope", "language"],
        how="left",
    ).drop_duplicates(subset=["start", "end", "transcription", "full_audio_file"])
    full_corpus_df.loc[:, "turn_speaker"] = full_corpus_df.speaker_id

    # Make sentence ids
    full_corpus_df["sentence_id"] = full_corpus_df.index

    # Handle missing values
    full_corpus_df["speaker_id"] = full_corpus_df["speaker_id"].fillna("unknown")
    full_corpus_df["language"] = full_corpus_df["language"].fillna("other")

    return full_corpus_df


def row_to_consolidated(row, audio_dir):
    dataset_prefix = "rundkast_"
    return consolidated_utterance(
        dataset_prefix + str(row["speaker_id"]),
        row["type"],
        dataset_prefix + str(row["sentence_id"]),
        row["language"],
        row["transcription"],
        str(Path(audio_dir) / row["full_audio_file"]),
        "rundkast",
        row["accent"] if type(row["accent"]) is str else "unknown",
        row["duration"],
        row["start"],
        row["end"],
    )


def parse_rundkast(rundkastdir):
    """Parse Rundkast files and return a list of consolidated utterances"""

    audiodir = Path(rundkastdir) / "audio"

    df = parse_corpus_files(rundkastdir)
    return list(df.apply(lambda row: row_to_consolidated(row, audiodir), axis=1))


def parse_rundkast_phon(rundkast_phon_dir):

    audiodir = Path(rundkast_phon_dir) / "audio"
    transdir = Path(rundkast_phon_dir) / "transcription"

    dataset_prefix = "rundkast_"

    segment_list = []
    for transfile in transdir.glob("*.TextGrid"):
        stem = transfile.stem
        audiofilename = stem + ".wav"
        audiofilepath = audiodir / audiofilename

        speaker_id = stem.split("_")[4]

        tg = TextGrid(transfile)
        utterance_ints = list(tg.get_tier("utterance").get_intervals())
        start = float(0)
        end = utterance_ints[-1][1]
        text = " ".join([utt[2] for utt in utterance_ints if utt[2] != "..."])
        wordlist = [
            {"word": x[2], "start": x[0], "end": x[1]}
            for x in tg.get_tier("word").get_intervals()
            if x[2] != "..."
        ]
        phonelist = [
            {"phone": x[2], "start": x[0], "end": x[1]}
            for x in tg.get_tier("phoneme").get_intervals()
            if x[2] != "..."
        ]
        segment_list.append(
            consolidated_utterance_phon(
                dataset_prefix + speaker_id,
                phon_speakers[speaker_id][0],
                dataset_prefix + stem,
                "nb-NO",
                text,
                wordlist,
                phonelist,
                str(audiofilepath),
                "rundkast_phon",
                phon_speakers[speaker_id][1],
                end,
                start,
                end,
            )
        )
    return segment_list
