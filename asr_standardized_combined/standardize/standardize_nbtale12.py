import os
import re

import random  # to show a few random transcripts
import datetime  # to store date of creation in config file
import json  # to create config file
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

sys.path.append("..")  # for importing from other dir
from ..parsers.nbtale_trans_parser import parse_nbtale

# set defaults here for parameters so we can use them between both the argparse and the standardize()
default_keep_nv = True
default_keep_v = True
default_annotation_token = "<"
default_keep_symbols = True
default_keep_numerals = True
default_keep_empty = True
default_verbose = False

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
    "-knva",
    "--keep_nv_annotations",
    action="store_{}".format(str(default_keep_nv).lower()),
    help="Keep non-verbal annotations",
)
parser.add_argument(
    "-kva",
    "--keep_v_annotations",
    action="store_{}".format(str(default_keep_v).lower()),
    help="Keep verbal annotations",
)
parser.add_argument(
    "-at",
    "--annotation_token",
    type=str,
    default=default_annotation_token,
    help="Token used in data to mark annotations",
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

# Variables that can change according to users' needs and corpora - In order of substitution
alphabet = "a b c d e f g h i j k l m n o p q r s t u v w x y z å ø æ - é –".split()


wordDic_num = {
    "e6": "e seks",
    "e18": "e atten",
    "u2s": "ju tus",
    "7-up": "seven øpp",
    "skal0": "skal",
    "3g-": "tre-g-",
    "7": "sju",
    "8": "åtte",
    "co2-utslippene": "se-o-to-utslippene",
}

wordDic_annot = {
    "<comma>": "",
    "<sil>": "",
    "<titlestart>": "",
    "<titleend>": "",
    "<anonymized>": "",
    "<sentenceboundary>": "",
}

wordDic_sounds = {
    "<fp>": "",
    "<inhale>": "",
    "<exhale>": "",
    "<vowel>": "eee",
    "<nasal>": "mmm",
    "<incomprehensible>": "",
}

wordDic_sym = {
    "–": "-",
    ",": " ",
    ".no": "dot no",
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
    "\x7f": "",
    '"': "",
    ";": "",
    "?": "",
    "!": "",
    "â": "a",
    "ë": "e",
    "<": "",
    "`": "",
}


def standardize(
    transcription_list,
    segmented_audio_list=[],
    keep_nv_annotations=default_keep_nv,
    keep_v_annotations=default_keep_v,
    annotation_token=default_annotation_token,
    keep_symbols=default_keep_symbols,
    keep_numerals=default_keep_numerals,
    keep_empty=default_keep_empty,
    verbose=default_verbose,
):

    """
    See documentation of functions involved for functionality and parameters.

    Parameters
    ----------
    keep_symbols: bool
        Determines whether we want to replace tokens according to wordDic_sym
    keep_numerals: bool
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

    # Numerals
    if not keep_numerals:
        standardized_transcripts = [
            replace_symbols(s, wordDic_num) for s in standardized_transcripts
        ]

    # Events without an associated sound but that modify the speech form
    if not keep_nv_annotations:
        standardized_transcripts = [
            replace_symbols(s, wordDic_annot) for s in standardized_transcripts
        ]

    # Events that could in principle have a transcription (not provided)
    if not keep_v_annotations:
        standardized_transcripts = [
            replace_symbols(s, wordDic_sounds) for s in standardized_transcripts
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

    if not keep_empty:
        # Remove empty utterances
        standardized_audios, standardized_transcripts = remove_empty_utt(
            standardized_transcripts, segmented_audio_list, annotation_token, alphabet
        )
    else:
        # set standardized_audios so we can return it later
        standardized_audios = segmented_audio_list

    # Checks
    if keep_empty:
        assert len(transcription_list) == len(
            standardized_transcripts
        ), """Something went wrong during the standardization, the original transcription
        list contained {} lines while the standardized transcripts list contains {} lines""".format(
            len(transcription_list), len(standardized_transcripts)
        )

    # Show a few random transcripts
    if verbose:
        logger.info("")
        logger.debug("Behold! See a few standardized sentences randomly selected:")
        for s in random.sample(standardized_transcripts, 10):
            logger.info("    {}".format(s))

    return standardized_transcripts, standardized_audios


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
    logger.info("Remove verbal annotations: {}".format(not args.keep_v_annotations))
    logger.info(
        "Remove non-verbal annotations: {}".format(not args.keep_nv_annotations)
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
        "Remove transcriptions that only contain annotations: {}".format(
            not args.keep_empty
        )
    )

    # Getting data - only free speech
    output = parse_nbtale(args.data_dir)
    audio_list, trans_list = zip(
        *[
            (o.audio_file, o.sentence_text_raw)
            for o in output
            if not "free" in o.audio_file
        ]
    )

    # Put in pandas dataframe and filter to only non "free" speech
    df = pd.DataFrame([vars(o) for o in output])
    df = df[~df.audio_file.str.contains("free")]
    segmented_audio_list = list(df.segmented_audio_file)

    # Standardizing data
    standardized_transcripts, standardized_audios = standardize(
        transcription_list=trans_list,
        keep_nv_annotations=args.keep_nv_annotations,
        keep_v_annotations=args.keep_v_annotations,
        annotation_token=args.annotation_token,
        keep_symbols=args.keep_symbols,
        keep_numerals=args.keep_numerals,
        keep_empty=args.keep_empty,
        segmented_audio_list=segmented_audio_list,
        verbose=args.verbose,
    )

    # Get the right dataframe
    df = df[df.segmented_audio_file.isin(standardized_audios)]
    df["standardized_text"] = standardized_transcripts

    # Analyze audios with suspicious transcripts. NOTE: you need to create a check_list
    # print('')
    # print('***PLAYING AUDIOS WITH SUSPICIOUS TRANSCRIPTS***')
    # play_checks(check_list, standardized_transcripts, segmented_audio_list, args.data_dir)

    # Debugging and listening
    out_of_alphabet_info = out_of_alphabet(standardized_transcripts, alphabet)

    if out_of_alphabet_info[1] != []:
        logger.warning("")
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
        logger.info("")
        logger.debug("***OPENING AUDIO FILES WITH TOKENS OUT OF ALPHABET***")
        # NOTE we're moving the audio play construction outside of the play_audios() funct
        # TODO we need to create the audio list with full path(s) here.
        play_audios(
            standardized_transcripts,
            out_of_alphabet_info,
            df.segmented_audio_file,
            args.data_dir,
        )

    # Saving data
    save_csv(args, df, args.data_dir, args.save_filename)
