# Example: python3 standardize_npsc.py -d '/s2t_torch/datasets/NPSC_1_1' -l 'nb-NO' -sw -sa -st " " -v
# Example: python3 standardize_npsc.py -d /s2t_torch/datasets/NPSC_1_1 -l nn-NO -sf save_test

from dataclasses import astuple
import datetime  # to store date of creation in config file
from collections import Counter
import csv  # to store csv
import json  # to create config file
import os
import random  # to show a few random transcripts
import re
from subprocess import call  # for opening audios in VSCode

import argparse
import logging

# Project imports
from .utils import (
    out_of_alphabet,
    substitute_underscores,
    replace_symbols,
    play_audios,
    substitute_hesitations,
)
import sys

# Patches
from .patch_npsc import patch_dict, date_patch

sys.path.append("..")  # for importing from other dir
from ..parsers.npsc_parser import create_sentence, parse_npsc
from ..parsers.shared_classes import consolidated_utterance

# set defaults here for parameters so we can use them between both the argparse and the standardize()
default_standard_words = True
default_annotation_token = "<"
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
    help="Path to main directory with the raw data",
)
parser.add_argument(
    "-sw",
    "--standard_words",
    action="store_{}".format(str(default_standard_words).lower()),
    help="Standardize non-standard words",
)
parser.add_argument(
    "-at",
    "--annotation_token",
    type=str,
    default=default_annotation_token,
    help="Token used in data to mark annotations",
)
parser.add_argument(
    "-st",
    "--substitution_token",
    type=str,
    default=default_substitution_token,
    help='If a token is given, it will be substituted for annotations. Token CANNOT be "[Tt]rue" or "[Ff]alse". If no token provided then -kat will determine annotation standardiation.',
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
    "–": "-",
    ",": " ",
    ".no": "dot no",
    ".": " ",
    "'": " ",
    "/": " ",
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
    "lhbt+": "lhtb-pluss",
    "omsorg+": "omsorg-pluss",
}

wordDic_num = {
    "4h": "fire h",
    "ks2": "ks to",
    "co2-": "co to ",
    "co2": "co to",
    "r5": "r fem",
    "r6": "r seks",
    "e18": "e atten",
    "e24": "e tjuefire",
    "e16": "e seksten",
    "e69": "e sekstini",
    "e6": "e seks",
    "3d": "tre d",
    "8-forslag": "åtte-forslag",
    "no2-": "no to ",
    "no2": "no to",
    "08": "null åtte",
    "5g": "fem g",
    "4g-": "fire g ",
    "4g": "fire g",
    "e39": "e trettini",
    "g20": "g tjue",
    "tek17": "tek sytten",
    "tek10": "tek ti",
    "a4": "a fire",
    "tv2": "tv to",
    "tv 2": "tv to",
    "g7": "g sju",
    "cop23": "cop tjuetre",
    "k5": "k fem",
    "k0": "k null",
    "u-864": "u åttehundre sekstifire",
    "f-35": "f-femogtredve",
    "strontium-90": "strontium nitti",
    "ks1": "ks én",
    "forny2020": "forny tjuetjue",
    "1g": "én g",
    "helseomsorg21-strategien": "helseomsorg-tjueén-strategien",
    "lektor2-programmet": "lektor-to-programmet",
    "artikkel-5-situasjon": "artikkel-fem-situasjon",
    "a4-løsninger": "a-fire-løsninger",
    "el6": "el-seks",
    "cv90-investeringene": "cv-nitti-investeringene",
    "nh90-": "nh nitti ",
    "nh90": "nh nitti",
    "ip3-mannskapene": "ip-tre-mannskapene",
    "ip3": "ip tre",
    "vg1": "vg én",
    "8-forslagene": "åtte-forslagene",
    "cop24": "cop-tjuefire",
    "p-3": "p-tre",
    "8-forslaget": "åtte-forslaget",
    "1000": "tusen",
    "365": "tre seks fem",
    "8:32": "åtte trettito",
    "2020": "tjuetjue",
    "2016": "tjueseksten",
    "2018": "tjueatten",
    "2014": "tjuefjorten",
    "2000": "to tusen",
    "25": "tjuefem",
    "4": "fire",
    "2": "to",
    "0": "null",
    "1": "én",
    "3": "tre",
}

wordDic_hes = {
    "<mm>": " mmm ",
    "<qq>": " qqq ",
    "<ee>": " eee ",
    "<inaudible>": " ",
}


def substitute_piped_words(sentence, standard_words):
    """
    For a sentence, identifies pairs of non-standard|standardized tokens and
    replaces them with either the non-standard or the standardized version.

    Parameters
    ----------
    sentence: str
    standard_words: bool
        Determines whether we want to choose the (non-)standard version
        (default is True, i.e. we use the standardized version)

    Returns
    -------
    sentence: str
        Sentence where the (non-)standard version of the words is chosen
    """
    tokenized_sentence = sentence.split()
    standardized_tokens = []
    for word in tokenized_sentence:
        if "|" in word:
            if "|1" in word:
                replaced_word = word.split("|")[0]
            else:
                if standard_words:
                    replaced_word = word.split("|")[1]
                else:
                    replaced_word = word.split("|")[0]
        else:
            replaced_word = word
        standardized_tokens.append(replaced_word)
    return " ".join(standardized_tokens)


def standardize(
    transcription_list,
    audio_list=[],
    standard_words=default_standard_words,
    keep_annotations=default_keep_annotations,
    annotation_token=default_annotation_token,
    substitution_token=default_substitution_token,
    keep_symbols=default_keep_symbols,
    keep_numerals=default_keep_numerals,
    keep_empty=default_keep_empty,
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

    if substitution_token.strip().lower() == "false":
        substitution_token = False

    del_letters, del_words, del_sentences, index_sentence = out_of_alphabet(
        transcription_list, alphabet=alphabet
    )

    # Non-standard words
    standardized_sentences = [
        substitute_piped_words(sentence, standard_words)
        if "|" in sentence
        else sentence
        for sentence in transcription_list
    ]

    if verbose:
        logger.setLevel(logging.DEBUG)

    nonstandard_piped_sentences = [
        sentence for sentence in del_sentences if "|" in sentence
    ]
    logger.debug("")
    logger.debug("***STANDARDIZATION OF NON-STANDARD WORDS***")
    if standard_words:
        logger.debug(
            "Number of sentences with non-standard words standardized: {}".format(
                len(nonstandard_piped_sentences)
            )
        )  # checks
    else:
        logger.debug(
            "Number of sentences with non-standard words NOT standardized: {}".format(
                len(nonstandard_piped_sentences)
            )
        )  # checks

    # Underscores
    underscored_sentences_after = [
        sentence for sentence in standardized_sentences if "_" in sentence
    ]
    logger.debug("")
    logger.debug("***REMOVING UNDERSCORES***")
    logger.debug(
        "Number of unique sentences with underscored words: {}".format(
            len(underscored_sentences_after)
        )
    )

    standardized_sentences = [
        substitute_underscores(sentence) if "_" in sentence else sentence
        for sentence in standardized_sentences
    ]

    # Lower case
    # TODO: adapt to lower_case=False (requires expanding wordDic_sym and wordDic_num among other things)
    # if lower_case:
    standardized_sentences = [s.lower() for s in standardized_sentences]

    # Non-verbal annotations
    hesitation_sentences = [
        sentence for sentence in standardized_sentences if annotation_token in sentence
    ]
    hesitation_words = [
        word
        for sentence in del_sentences
        for word in sentence.lower().split()
        if annotation_token in word
    ]
    logger.debug("")
    logger.debug("***NON-VERBAL ANNOTATIONS***")
    logger.debug(
        "Number of unique sentences with annotations: {}".format(
            len(hesitation_sentences)
        )
    )
    if substitution_token:
        logger.debug(
            'Different annotations found in the data substituted by "{}": {}'.format(
                substitution_token, dict(Counter(hesitation_words))
            )
        )
    else:
        if keep_annotations:
            logger.debug(
                "Different annotations found in the data and left unchanged: {}".format(
                    dict(Counter(hesitation_words))
                )
            )
        else:
            logger.debug(
                "Different annotations found in the data changed to triple-letter format and with the token removed (inaudible removed): {}".format(
                    dict(Counter(hesitation_words))
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
            replace_symbols(s, wordDic_sym) for s in standardized_sentences
        ]

    # Numerals
    if not keep_numerals:
        standardized_sentences = [
            replace_symbols(s, wordDic_num) for s in standardized_sentences
        ]

    # Remove useless spaces
    standardized_sentences = [re.sub(" +", " ", s) for s in standardized_sentences]
    standardized_sentences = [s.strip() for s in standardized_sentences]

    # Remove empty utterances or those containing just a non-verbal annotation
    standardized_audios = audio_list
    if not keep_empty:
        hesitation_words = [
            word
            for sentence in del_sentences
            for word in sentence.lower().split()
            if annotation_token in word
        ]
        if not keep_annotations:
            filters = ["", " ", substitution_token]
        if keep_annotations and substitution_token:
            filters = ["", " "] + list(set(hesitation_words))
            filters = list(
                set(
                    [
                        f.replace("<mm>", "mmm")
                        .replace("<ee>", "eee")
                        .replace("<qq>", "qqq")
                        .replace("<inaudible>", "")
                        for f in filters
                    ]
                )
            )  # TODO: use replace_symbols
        if keep_annotations and not substitution_token:
            filters = ["", " "] + list(set(hesitation_words))
        logger.debug("")
        logger.debug(
            "Removing utterances that only consist of one of the following: {}".format(
                filters
            )
        )
        standardized_audios, standardized_sentences = zip(
            *[
                (a, s)
                for (a, s) in zip(standardized_audios, standardized_sentences)
                if s not in filters
            ]
        )

    # Checks
    if keep_empty:
        assert len(transcription_list) == len(
            standardized_sentences
        ), """Something went wrong during the standardization, the original transcription
        list contained {} lines while the standardized transcripts list contains {} lines""".format(
            len(transcription_list), len(standardized_sentences)
        )
    else:
        logger.info(
            "The original list contained {} utterances while the standardized list contains {} utterances".format(
                len(transcription_list), len(standardized_sentences)
            )
        )

    # Show a few random transcripts
    if verbose:
        logger.debug("")
        logger.debug("Behold! See a few standardized sentences randomly selected:")
        for s in random.sample(standardized_sentences, 10):
            logger.info("    {}".format(s))

    return standardized_sentences, standardized_audios


def save_csv(
    args, consolidated_utterances, transcription_list, audio_list, data_dir, filename
):
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

        # TODO: optimize the csv writing, it's quite slow. Maybe creating a pandas dataframe first and then filtering is quicker
        with open("{}.csv".format(stamped_path_to_filename), "w") as stream:
            writer = csv.writer(stream)
            for c in consolidated_utterances:
                row = astuple(c)
                if c.segmented_audio_file in audio_list:
                    row += (
                        transcription_list[audio_list.index(c.segmented_audio_file)],
                    )
                    writer.writerow(row)

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
    from asr_standardized_combined.standardize.__init__ import create_new_logger
    logger = create_new_logger(logging.getLogger(__name__), __file__)
    args = parser.parse_args()

    # Options chosen
    logger.info("Standardizing data from {}".format(args.data_dir))
    logger.info("Language selected: {}".format(args.language))
    logger.info("Standardize non-standard words: {}".format(args.standard_words))
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
    output = parse_npsc(args.data_dir)
    if args.language == "both":
        audio_list, trans_list = zip(
            *[
                (o.segmented_audio_file, o.sentence_text_raw)
                for o in output
                if o.sentence_language_code != "en-US"
            ]
        )
    else:
        audio_list, trans_list = zip(
            *[
                (o.segmented_audio_file, o.sentence_text_raw)
                for o in output
                if o.sentence_language_code == args.language
            ]
        )

    # Standardizing data
    standardized_transcripts, standardized_audios = standardize(
        trans_list,
        standard_words=args.standard_words,
        annotation_token=args.annotation_token,
        substitution_token=args.substitution_token,
        keep_annotations=args.keep_annotations,
        keep_symbols=args.keep_symbols,
        keep_numerals=args.keep_numerals,
        keep_empty=args.keep_empty,
        audio_list=audio_list,
        verbose=args.verbose,
    )

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
        logger.info("")
        logger.debug("***OPENING AUDIO FILES WITH TOKENS OUT OF ALPHABET***")
        play_audio_list = [
            os.path.join(args.data_dir, audio_item[:8], "audio", audio_item)
            for audio_item in audio_list
        ]
        play_audios(standardized_transcripts, out_of_alphabet_info, play_audio_list)

    # Saving data
    save_csv(
        args,
        output,
        standardized_transcripts,
        standardized_audios,
        args.data_dir,
        args.save_filename,
    )
