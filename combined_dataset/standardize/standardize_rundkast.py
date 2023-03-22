# Example: python3 standardize_npsc.py -d '/s2t_torch/datasets/NPSC_1_1' -l 'nb-NO' -sw -sa -st " " -v
# Example: python3 standardize_npsc.py -d /s2t_torch/datasets/NPSC_1_1 -l nn-NO -sf save_test

from dataclasses import astuple
import datetime  # to store date of creation in config file
from collections import Counter
import csv  # to store csv
import json  # to create config file
import os
import pandas as pd
import random  # to show a few random transcripts
import re
from subprocess import call  # for opening audios in VSCode
from pydub import AudioSegment  # to segment the audio

import argparse
import logging

# Project imports
from .utils import (
    out_of_alphabet,
    replace_symbols,
    export_audio_segments,
    substitute_hesitations,
)
import sys

# Patches
from .patch_rundkast import patch_dict, date_patch

sys.path.append("..")  # for importing from other dir
from ..parsers.rundkast_parser import parse_rundkast

# default_keep_annotations = True
default_annotation_token = "["
default_substitution_token = "False"
default_keep_annotations = True
default_keep_symbols = True
default_keep_numerals = True
default_keep_empty = True
default_verbose = False
default_patch = True


# Parser
parser = argparse.ArgumentParser(
    description="Standardize transcriptions for ASR training and testing"
)
parser.add_argument(
    "-l",
    "--language",
    type=str,
    default="both",
    choices=["nb-NO", "nn-NO", "both"],
    help="Language to select",
)
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    required=True,
    help="Path to main Rundkast directory.",
)
parser.add_argument(
    "-at",
    "--annotation_token",
    type=str,
    default=default_annotation_token,  # "[",
    help="Token used in data to mark annotations",
)
parser.add_argument(
    "-st",
    "--substitution_token",
    type=str,
    default=default_substitution_token,
    help="Token used to substitute annotations when -sa given",
)
parser.add_argument(
    "-kat",
    "--keep_annotations",  # _token",
    action="store_{}".format(str(default_keep_annotations).lower()),
    help="Keep the orignal annotations. If False, annotations will be substituted according to a hard-coded dictionary.",
)
parser.add_argument(
    "-ks",
    "--keep_symbols",
    action="store_{}".format(str(default_keep_symbols).lower()),
    help="Keep not alphanumeric symbols outside the alphabet",
)
parser.add_argument(
    "-kn",
    "--keep_numerals",
    action="store_{}".format(str(default_keep_numerals).lower()),
    help="Keep numerals (otherwise substitute them by their orthographic version)",
)
parser.add_argument(
    "-ke",
    "--keep_empty",
    action="store_{}".format(str(default_keep_empty).lower()),
    help="Keep transcriptions only containing non-verbal sounds",
)
parser.add_argument(
    "-sf",
    "--save_filename",
    type=str,
    default=None,
    help="""Saves a csv including standardized transcriptions in data_dir/standardized_csvs/save_filename.csv
                    and a config json file data_dir/standardized_csvs/save_filename.json with the configuration chosen""",
)
parser.add_argument(
    "-li",
    "--listen",
    action="store_true",
    help="Opens audio files in VSCode whose transcriptions contain tokens out of alphabet after standardization",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_{}".format(str(default_verbose).lower()),
    help="Displays debugging information",
)

# Logger
# grab the logger that we set up in the init file
logger = logging.getLogger(__name__)

# Variables that can change according to users' needs and corpora
alphabet = "a b c d e f g h i j k l m n o p q r s t u v w x y z å ø æ - é –".split()

wordDic_sym = {
    # punct
    "–": "-",
    # "_": " ", # commented out to allow "_" as substitution token.
    ",": " ",
    ".": " ",
    '"': " ",
    "`": "'",
    "\\": "'",
    "'s": "s", # added this for the correct Norwegian form for possessives
    "'": " ",
    "/": " ",
    "*": " ",
    "?": " ",
    "!": " ",
    "|": " ",
    ";": " ",
    ":": " ",
    "+": " ",
    # letters
    "è": "e",
    "ê": "e",
    "ö": "ø",
    "ò": "o",
    "ó": "o",
    "ô": "o",
    "ä": "a",
    "à": "a",
    "ï": "i",
    "í": "i",
    "ü": "y",
    "ç": "s",
}

wordDic_num = {
    "16": " seksten ",
    "p 1": " p én ",
    "p 2": " p to ",
    "2": " to ",
    "300": " tre hundre ",
    "3": " tre ",
    "4": " fire ",
    "e 6": " e seks ",
}

wordDic_hes = {
    "[m]": " mmm ",
    "[s]": " qqq ",
    "[e]": " eee ",
}

annotation_pattern = re.compile("\[.*?\]")


def audio_path(df, path_to_data):
    """
    Given the path to downloaded data and a dataframe with the audio file names and start and end times,
    obtain the path to the full audio file and the sentence-segmented ones in your system. This is used
    to create later on the sentence-segmented audio files using the path specified below.

    Parameters
    ----------
    df: pandas Dataframe
        Dataframe inherited from the consolidated_utterance class. Among its columns, we need to have
        audio_file (name of the full speech file), start_time, and end_time (start and end times in seconds
        of the sentence segments, respectively).
    path_to_data: str
        Path to the main directory where the data is stored.

    Returns
    -------
    path_segment, path_total: tuple of strings
        Path in your system to the full audio file and the sentence-segmented files.
    """
    start = round(df.start_time * 1000)
    end = round(df.end_time * 1000)
    path_total = os.path.join(path_to_data, df.audio_file)
    audio = df.audio_file.split("/")[-1].split(".")[0]
    path_segment = os.path.join(
        path_to_data,
        "audio_segments",
        "{}_{}_{}.wav".format(audio, start, end),
    )
    return path_segment, path_total


def standardize(
    transcription_list,
    keep_annotations=default_keep_annotations,
    annotation_token=default_annotation_token,
    substitution_token=default_substitution_token,
    keep_symbols=default_keep_symbols,
    keep_numerals=default_keep_numerals,
    verbose=default_verbose,
    patch=default_patch,
):

    """
    See documentation of functions involved for functionality and parameters.

    Parameters
    ----------
    substitute_symbols: bool
        Determines whether we want to replace tokens according to wordDic_sym
    substitute_numerals: bool
        Determines whether we want to replace tokens according to wordDic_num
    verbose: bool
        Outputs extra information about the standardization process (default is True)

    Returns
    -------
    standardized_audios, standardized_sentences: tuple of lists
        Lists of standardized audio filenames and sentences
    """

    ### Things to expect in a Rundkast transcription ###
    # - Capital letters of names and acronyns. E.g. "...fra Oslo og..."
    # - Punctuation: normal, expected punctuation is . , ? but others sneak in
    # - Hyphen: used in some unusual compound words. E.g. "påske-utfarten"
    # - Apostrophe: used between proper names/acronyms and affixes. E.g. "...i ordføreren's bil..."
    # - Parentheses: used to mark truncated words E.g. "...for d()så jeg..."
    # - Underscores: used to mark words forms not allowed by the standard of the speaker. E.g. "_itj"
    # - Square brackets: mark event tags, such things as breathing, pauses, hesitations, and mispronunciations. E.g. "...starter [e] påske..."
    # - Astrix: marks mispronunciations. Can be used both inside square brackets as well as before individual words.  E.g. "[*-] ikke [-*]" and "*ikke"
    ####################################################


    if verbose:
        logger.setLevel(logging.DEBUG)

    if substitution_token.strip().lower() == "false":
        logger.debug("substitution_token has been set to the bool False")
        substitution_token = False
    else:
        logger.debug(
            "substitution_token.strip().lower() is --{}--".format(
                substitution_token.strip().lower()
            )
        )

    # Lower case
    standardized_sentences = [s.lower() for s in transcription_list]

    logger.debug(
        "Sub_token is {} which is a {}".format(
            substitution_token, type(substitution_token)
        )
    )
    standardized_sentences = [
        substitute_hesitations(
            sentence,
            keep_annotations,
            annotation_token,
            substitution_token,
            wordDic_hes,
        )
        if annotation_token in sentence
        else sentence
        for sentence in standardized_sentences
    ]

    # Patch
    if patch:
        logger.info("Applying patch dated {}".format(date_patch))
        standardized_sentences = [
            replace_symbols(s, patch_dict) for s in standardized_sentences
        ]
    else: logger.info("NOT applying patch")
    
    # Symbols
    if not keep_symbols:
        standardized_sentences = [
            replace_symbols(s, wordDic_sym)
            for s in standardized_sentences
        ]
        underscore_pattern_start = re.compile('_(\w)')
        underscore_pattern_end = re.compile('(\w)_')
        standardized_sentences = [
            underscore_pattern_start.sub(
                '\g<1>',
                underscore_pattern_end.sub(
                    '\g<1>',
                    s
                )
            )
            for s in standardized_sentences  
        ]

    # Numerals
    if not keep_numerals:
        standardized_sentences = [
            replace_symbols(s, wordDic_num) for s in standardized_sentences
        ]

    # Remove parentheses
    standardized_sentences = [
        sentence.replace("()", " ").replace('(', '').replace(')','') for sentence in standardized_sentences
    ]

    # Remove useless spaces
    standardized_sentences = [re.sub(" +", " ", s) for s in standardized_sentences]
    standardized_sentences = [s.strip() for s in standardized_sentences]

    # Checks
    assert len(transcription_list) == len(
        standardized_sentences
    ), """Something went wrong during the standardization, the original transcription
    list contained {} lines while the standardized transcripts list contains {} lines""".format(
        len(transcription_list), len(standardized_sentences)
    )

    # Show a few random transcripts
    if verbose:
        logger.debug("")
        logger.debug("Behold! See a few standardized sentences randomly selected:")
        for s in random.sample(
            standardized_sentences,
            (10 if len(standardized_sentences) > 10 else len(standardized_sentences)),
        ):
            logger.debug("    %s", s)

    return standardized_sentences


def rundkast_annotation_to_blank(in_str):
    if annotation_pattern.match(in_str.strip()):
        return ""
    return in_str


def triple_letter_to_blank(in_str):
    if in_str in ["eee", "mmm", "qqq"]:
        return ""
    return in_str


def sub_token_to_blank(in_str, token):
    if in_str.strip() == token.strip():
        return ""
    return in_str


def save_csv(args, output_df, data_dir, filename):
    """
    Saves a csv file with the consolidated utterances and an extra column with the standardized transcriptions.
    If some transcriptions have been removed during the standardization process, the original utterances
    are also removed from the csv file. It also saves a json file with the standardization options chosen and date.

    Parameters
    ----------
    args: Namespace object
        Output of the argument parser, parser.parse_args()
    consolidated_utterances: list
        List of consolidated_utterance dataclass objects as output of parse_npsc.
    transcription_list: list of strings
        List of transcriptions that have been already standardized.
    audio_list: list of strings
        List of audio filenames associated to transcription_list (equal length).
    data_dir: str
        Path to the main directory where the data is stored.
    filename: str
        File name for the csv file.
        The arg parser has a default value of None, in which case data is not saved.

    Returns
    -------
    Nothing, it saves the data in "data_dir/standardized_csvs/filename.csv".
    """
    if filename is not None:
        stamp = datetime.datetime.now().strftime("%Y%m%d")
        logger.debug("")
        path_to_file = os.path.join(data_dir, "standardized_csvs")
        path_to_filename = os.path.join(path_to_file, filename)
        stamped_path_to_filename = "{}_{}".format(path_to_filename, stamp)
        if not os.path.exists(path_to_file):
            os.mkdir(path_to_file)

        # Don't overwrite - the choices below are rather arbitrary, feel free to suggest improvements
        if os.path.exists("{}.csv".format(stamped_path_to_filename)):
            for n in range(1, 100):
                stamped_path_to_filename = "{}_{}".format(path_to_filename, stamp)
                if os.path.exists("{}_{}.csv".format(stamped_path_to_filename, n)):
                    logger.info(
                        "File already exists, saving as {}_{}.csv instead".format(
                            stamped_path_to_filename, n + 1
                        )
                    )
                    stamped_path_to_filename = "{}_{}".format(
                        stamped_path_to_filename, n + 1
                    )
                else:
                    stamped_path_to_filename = "{}_{}".format(
                        stamped_path_to_filename, n
                    )
                    break

        logger.info("Saving csv to {}.csv".format(stamped_path_to_filename))

        # TODO: optimize the csv writing, it's quite slow. Maybe creating a pandas dataframe first and then filtering is quicker
        output_df.to_csv(
            "{}.csv".format(stamped_path_to_filename), header=False, index=False
        )

        # TODO: Warn when properties in json file coincide (other than csv_creation_date)

        # Dump to json
        config_dict = vars(args)
        config_dict["csv_creation_date"] = stamp
        config_dict = {
            k: v
            for k, v in config_dict.items()
            if k not in ["listen", "save_filename", "verbose"]
        }
        logger.info("Saving config to {}.json".format(stamped_path_to_filename))
        with open("{}.json".format(stamped_path_to_filename), "w") as fp:
            json.dump(config_dict, fp)


if __name__ == "__main__":
    from combined_dataset.standardize.__init__ import create_new_logger

    logger = create_new_logger(logging.getLogger(__name__), __file__)

    args = parser.parse_args()

    # Options chosen
    logger.info("Standardizing data from {}".format(args.data_dir))
    logger.info("Language selected: {}".format(args.language))
    
    if not args.keep_annotations:
        logger.info(
            "Keeping annotations (except <inaudible>) in triple-letter format but removing marking tokens"
        )
    logger.info(
        "Substitute non-alphanumeric symbols outside the alphabet: {}".format(
            not args.keep_symbols
        )
    )
    logger.info(
        "Substitute numerals by their orthographic version: {}".format(
            not args.keep_numerals
        )
    )
    logger.info(
        "Remove transcriptions that only contain a single non-verbal sound: {}".format(
            not args.keep_empty
        )
    )

    # Getting data
    transdir = "../data/rundkast"
    output = pd.DataFrame(parse_rundkast(args.data_dir))
    if args.language == "both":
        output = output[output["sentence_language_code"] != "en-US"]
    else:
        output = output[output["sentence_language_code"] == args.language]
    trans_list = list(output["sentence_text_raw"])

    # Sentence-segmented audio files
    print("")
    print("***SENTENCE-SEGMENTED AUDIO***")
    output["segmented_audio_file"] = output.apply(
        lambda row: audio_path(row, args.data_dir)[0], axis=1
    )
    path_to_segment_dir = os.path.join(args.data_dir, "audio_segments")
    logger.info("Sentence-segmented audio files in {}".format(path_to_segment_dir))
    output.apply(
        lambda row: export_audio_segments(row, *audio_path(row, args.data_dir)), axis=1
    )
    segmented_audio_list = output.segmented_audio_file.tolist()
    logger.info("Total number of segments: {}".format(output.shape[0]))

    # Standardizing data
    standardized_transcripts = standardize(
        trans_list,
        keep_annotations=args.keep_annotations,
        annotation_token=args.annotation_token,
        substitution_token=args.substitution_token,
        keep_symbols=args.keep_symbols,
        keep_numerals=args.keep_numerals,
        verbose=args.verbose,
    )

    output.loc[:, "standardized_transcripts"] = standardized_transcripts

    if not args.keep_empty:
        if args.substitution_token:
            # all annotations will have been replaced with token
            output.loc[
                :, "standardized_transcripts"
            ] = output.standardized_transcripts.apply(
                lambda x: sub_token_to_blank(x, args.substitution_token)
            )
        elif not args.keep_annotations:
            # we've either deleted the annotation or replaces with eee/mmm/qqq
            output.loc[
                :, "standardized_transcripts"
            ] = output.standardized_transcripts.apply(
                lambda x: triple_letter_to_blank(x)
            )
        else:
            # no annotation subs have been done
            output.loc[
                :, "standardized_transcripts"
            ] = output.standardized_transcripts.apply(
                lambda x: rundkast_annotation_to_blank(x)
            )
        output = output[output.standardized_transcripts != ""]

    # Saving data
    save_csv(args, output, args.data_dir, args.save_filename)
