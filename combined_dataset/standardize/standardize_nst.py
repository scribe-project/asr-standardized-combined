import os
import re
from subprocess import call  # for opening audios in VSCode
import random  # to show a few random transcripts
import datetime  # to store date of creation in config file
import json
from tabnanny import verbose  # to create config file
import pandas as pd

# Ignore the pydub warning - ffmpeg is there and it works
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


warnings.filterwarnings("default", category=RuntimeWarning)

import argparse
import logging

# Project imports
from .utils import (
    out_of_alphabet,
    substitute_underscores,
    replace_symbols,
    remove_empty_utt,
    play_audios,
)
import sys

# Patches
from .patch_nst import patch_dict, date_patch

sys.path.append("..")  # for importing from other dir
from ..parsers.nst_parser import parse_nst

# set defaults here for parameters so we can use them between both the argparse and the standardize()
default_keep_symbols = True
default_keep_numerals = True
default_verbose = False
default_patch = True

# Parser
parser = argparse.ArgumentParser(
    description="Standardize transcriptions for ASR training and testing"
)
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    required=True,
    help="Path to main directory with the raw data",
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

# Variables that can change according to users' needs and corpora - In order of substitution
alphabet = "a b c d e f g h i j k l m n o p q r s t u v w x y z å ø æ - é –".split()

wordDic_sym = {
    "–": "-",
    ",": " ",
    ".": " ",
    "'": "",
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
    " ca ": " cirka ",
    ";": "",
    "?": "",
    "!": "",
    "â": "a",
    "á": "a",
    "+": "",
    "\\": " ",
    "/": " ",
    "(": "",
    ")": "",
}

wordDic_num = {
    "398": "tre hunde og nittiåtte",
    "397": "tre hunde og nittisju",
}


def standardize(
    transcription_list,
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
        Outputs extra information about the standardization process (default is False)

    Returns
    -------
    standardized_audios, standardized_sentences: tuple of lists
        Lists of standardized audio filenames and sentences
    """

    if verbose:
        logger.setLevel(logging.DEBUG)

    # Lower case and substitutions needed to be able to parse the transcriptions
    standardized_transcripts = [s.lower() for s in transcription_list]

    # Patch
    if patch:
        logger.info("Applying patch dated {}".format(date_patch))
        standardized_transcripts = [
            replace_symbols(s, patch_dict) for s in standardized_transcripts
        ]
    else: logger.info("NOT applying patch")
        
    # Numerals
    if not keep_numerals:
        standardized_transcripts = [
            replace_symbols(s, wordDic_num) for s in standardized_transcripts
        ]

    # Special characters and abbreviations
    if not keep_symbols:
        standardized_transcripts = [
            replace_symbols(s, wordDic_sym) for s in standardized_transcripts
        ]

    # Underscores (important: after normalizations)
    standardized_transcripts = [
        substitute_underscores(sentence) if "_" in sentence else sentence
        for sentence in standardized_transcripts
    ]

    # Remove useless spaces
    standardized_transcripts = [re.sub(" +", " ", s) for s in standardized_transcripts]
    standardized_transcripts = [s.strip() for s in standardized_transcripts]

    # Show a few random transcripts
    if verbose:
        logger.info("")
        logger.debug("Behold! See a few standardized sentences randomly selected:")
        for s in random.sample(standardized_transcripts, 10 if len(standardized_transcripts) >= 10 else len(standardized_transcripts)):
            logger.info("    {}".format(s))

    return standardized_transcripts


def save_csv(args, df, data_dir, filename):
    """
    Saves a csv file with the consolidated utterances and two extra columns with the sentence-segmented audio files'
    path and the standardized transcriptions. If some transcriptions have been removed during the standardization
    process, the original utterances are also removed from the csv file. It also saves a json file with the
    standardization options chosen and date.

    Parameters
    ----------
    args: Namespace object
        Output of the argument parser, parser.parse_args()
    df: pandas Dataframe
        Pandas dataframe containing the data after cleaning.
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
        logger.info("")
        path_to_file = os.path.join(data_dir, "standardized_csvs")
        path_to_filename = os.path.join(path_to_file, filename)
        stamped_path_to_filename = "{}_{}".format(path_to_filename, stamp)
        if not os.path.exists(path_to_file):
            os.mkdir(path_to_file)

        # Don't overwrite - the choices below are rather arbitrary
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

        df.to_csv("{}.csv".format(stamped_path_to_filename), header=False, index=False)

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

    # Parse arguments
    args = parser.parse_args()

    # Options chosen
    logger.info("Standardizing data from {}".format(args.data_dir))
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

    # Getting data
    output = parse_nst(args.data_dir, verbose=args.verbose)

    # Put in pandas dataframe and filter to only non "free" speech
    df = pd.DataFrame([vars(o) for o in output])
    trans_list = list(df.sentence_text_raw)

    # Standardizing data
    standardized_transcripts = standardize(
        transcription_list=trans_list,
        keep_symbols=args.keep_symbols,
        keep_numerals=args.keep_numerals,
        verbose=args.verbose,
    )

    # Get the right dataframe
    df["standardized_text"] = standardized_transcripts

    # Debugging and listening
    out_of_alphabet_info = out_of_alphabet(standardized_transcripts, alphabet)

    if out_of_alphabet_info[1] != []:
        logger.info("")
        logger.warning("***TRANSCRIPTS WITH TOKENS OUT OF ALPHABET***")
        logger.warning(
            "Tokens containing characters out of alphabet at this point: {}".format(
                out_of_alphabet_info[1]
            )
        )
        logger.warning(
            "Number of sentences with tokens out of alphabet: {}".format(
                len(out_of_alphabet_info[2])
            )
        )
    if len(out_of_alphabet_info[1]) > 1:
        logger.warning(
            "You have transcriptions with several tokens outside the alphabet you defined -- reconsider your (life) choices"
        )

    if args.listen and out_of_alphabet_info[1] != []:
        # Listen to audio files where tokens with characters out of the alphabet appear
        logger.debug("")
        logger.debug("***OPENING AUDIO FILES WITH TOKENS OUT OF ALPHABET***")
        # NOTE we're moving the audio play construction outside of the play_audios() funct
        # TODO we need to create the audio list with full path(s) here. I dunno what that's supposed to look like
        play_audios(
            standardized_transcripts,
            out_of_alphabet_info,
            df.segmented_audio_file,
            args.data_dir,
        )

    # Saving data
    save_csv(args, df, args.data_dir, args.save_filename)
