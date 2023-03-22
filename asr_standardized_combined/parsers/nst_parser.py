import os
import re
import sys
import json
from dataclasses import dataclass
import wave
import pandas as pd
from .shared_classes import consolidated_utterance

import logging

logger = logging.getLogger(__name__)

words_nn = [
    "ein",
    "dei",
    "ikkje",
    "eg",
    "eit",
    "me",
    "frå",
    "då",
    "noreg",
    "kva",
    "meir",
    "vere",
    "talar",
    "fleire",
    "gjer",
    "noko",
    "arbeidarpartiet",
    "korleis",
    "desse",
    "ver",
    "nokon",
    "gjere",
    "høgre",
    "vidare",
    "betre",
    "ifrå",
    "handlar",
    "kommunane",
    "utan",
    "røysta",
    "røystar",
    "vegner",
    "synest",
    "fleirtal",
    "kommunar",
    "sidan",
    "tidlegare",
    "kvar",
    "sit",
    "tydeleg",
    "éin",
    "eitt",
    "arbeidsplassar",
    "gong",
    "saman",
    "gjera",
    "treng",
    "veg",
    "millionar",
    "einig",
    "allereie",
    "pengar",
    "innanfor",
    "særleg",
    "eigentleg",
    "fekk",
    "kunna",
    "korfor",
    "einige",
    "krevjande",
    "eigen",
    "voterast",
    "viktigaste",
    "ressursane",
    "alvorleg",
    "vanskeleg",
    "fleirtalet",
    "utfordringar",
    "ressursar",
    "høgare",
    "tilstrekkeleg",
    "deira",
    "forbod",
    "sivilombodsmannen",
    "eigne",
    "reforma",
    "dårleg",
    "framstegspartiet",
    "finst",
    "eige",
    "aukar",
    "utanfor",
    "nokre",
    "verda",
    "delar",
    "politireforma",
    "snakkar",
    "gonger",
    "takkar",
    "innan",
    "difor",
    "sjølvsagt",
    "kvifor",
    "kome",
    "høyrer",
    "kristeleg",
    "eigarskap",
    "seinare",
    "håpar",
    "verkeleg",
    "saksordførar",
    "bidreg",
    "raudt",
    "innbyggarane",
    "antal",
    "ynskjer",
    "set",
    "annan",
    "vanlege",
    "løysingar",
    "milliardar",
    "dyrevelferda",
    "forventar",
    "verkemiddel",
    "høyring",
    "føreslå",
    "konsesjonslova",
    "bygga",
    "tenester",
    "noregs",
    "aktørar",
    "jobbar",
    "sterkare",
    "kven",
    "eine",
    "sikrar",
    "gruppehald",
    "berekraftig",
    "vesentleg",
    "betydeleg",
    "legga",
    "stortingsfleirtalet",
    "fornøgd",
    "usikkerheit",
    "pasientar",
    "nemleg",
    "høyrt",
    "val",
    "vedteke",
    "tener",
    "oppgåver",
    "hovudsak",
    "vurderingar",
    "offentleg",
    "endringane",
    "framleis",
    "takka",
    "endå",
    "halde",
    "utfordringane",
    "samarbeidspartia",
    "trengst",
    "forhandlingane",
    "kystsamfunna",
    "følgt",
    "verkar",
    "skapa",
    "verdiar",
    "offentlege",
    "investeringar",
    "einaste",
    "konsekvensane",
    "skikkeleg",
    "gjekk",
    "omfattande",
    "avgjerande",
    "talarstolen",
    "statlege",
    "svara",
    "tilbod",
    "lesa",
    "nord-noreg",
    "soldatar",
    "spørje",
    "brukar",
    "staden",
    "pasientane",
    "eigedomar",
    "konsekvensar",
    "faglege",
    "finna",
    "ytterlegare",
    "mogleg",
    "openbert",
    "eiga",
    "innbyggarar",
    "manglar",
    "innbyggjarane",
    "pengane",
    "såkalla",
    "reglar",
    "utanlandske",
    "personar",
    "sjølvstendig",
    "folkevalde",
    "seia",
    "støttar",
    "representantane",
    "koma",
    "vedtaka",
    "vedteken",
    "parkane",
    "brukast",
    "nemnde",
    "enklare",
    "utvalet",
    "tal",
    "stortingsrepresentantane",
    "naturleg",
    "forsvarleg",
    "fiskarar",
    "eigedom",
    "tilsvarande",
    "setta",
    "breitt",
    "gjerast",
    "låg",
    "fortsetta",
    "teke",
    "tala",
    "byane",
    "samstundes",
    "valt",
    "dårlegare",
    "næringar",
    "refererast",
    "regionalparkar",
    "politikarar",
    "aukande",
    "viktigare",
    "resultata",
]


def pred_nynorsk(sent, wl):
    lang = "nb-NO"
    for word in sent.lower().split(" "):
        if word in wl:
            lang = "nn-NO"
            break
    return lang


def get_audio_duration(filename):
    # returns: duration in seconds
    with wave.open(filename) as f_open:
        duration = f_open.getnframes() / f_open.getframerate()
    return duration


def get_nst_dialect(rob, roy):
    if rob != roy:
        return "unknown"
    elif rob in ["Oslo-området", "Ytre Oslofjord", "Hedmark og Oppland"]:
        return "east"
    elif rob in [
        "Sør-Vestlandet",
        "Bergen og Ytre Vestland",
        "Voss og omland",
        "Sunnmøre",
    ]:
        return "west"
    elif rob == "Sørlandet":
        return "south"
    elif rob == "Trøndelag":
        return "mid"
    elif rob in ["Nordland", "Troms"]:
        return "north"
    else:
        return "unknown"


def parse_nst(nst_path, channel="1", verbose=False):
    '''By default, the path to the audio from channel 1 is given.
    For channel 2, channel="2", and for stereo, channel="begge"'''
    datasets = ["ADB_NOR_0463", "ADB_NOR_0464"]
    audio_path = os.path.join(nst_path, f"lydfiler_16_{channel}/no/")
    final_results = []
    found_audio_files = 0
    missing_audio_files = 0
    dataset_prefix = "nst_"
    for dataset in datasets:
        for root, dirs, files in os.walk(os.path.join(nst_path, dataset)):
            for name in files:
                if name.endswith("json"):
                    file = os.path.join(root, name)
                    # print(f"processing json file: {file}")
                    with open(file, "r") as read_file:
                        data = json.load(read_file)
                    if "val_recordings" in data.keys():
                        speaker_id = data["info"]["Speaker_ID"]
                        sex = data["info"]["Sex"]
                        age = data["info"]["Age"]
                        pid = data["pid"]
                        region_of_birth = data["info"]["Region_of_Birth"]
                        region_of_youth = data["info"]["Region_of_Youth"]
                        df_full = pd.json_normalize(data, "val_recordings")
                        df = df_full.drop(
                            labels=[
                                "DST",
                                "NOI",
                                "QUA",
                                "SND",
                                "SPC",
                                "UTT",
                                "t0",
                                "t1",
                                "t2",
                                "type",
                            ],
                            axis="columns",
                        )
                        df["speaker_id"] = speaker_id
                        df["sex"] = sex
                        df["age"] = age
                        df["dataset"] = "NST"
                        sex = sex.lower() if sex in ["Female", "Male"] else "unknown"
                        path_list = []
                        wav_list = []
                        parsed_set = []
                        for row in df.itertuples():
                            file = row[1]  # wav file names
                            text = row[2]  # transcriptions
                            fn_raw = file.split(".")[0]
                            file_path = os.path.join(
                                audio_path, pid, pid + "_" + fn_raw + "-1.wav"
                            )
                            try:
                                found_audio_files += 1
                                path_list.append(file_path)
                                wav_list.append(fn_raw + "-1.wav")
                                duration = get_audio_duration(file_path)
                                parsed_set.append(
                                    consolidated_utterance(
                                        dataset_prefix + speaker_id,
                                        sex,
                                        dataset_prefix + fn_raw,
                                        pred_nynorsk(text, words_nn),
                                        text,
                                        file_path,
                                        dataset_prefix + "train"
                                        if dataset == "ADB_NOR_0463"
                                        else dataset_prefix + "test",
                                        get_nst_dialect(
                                            region_of_birth, region_of_youth
                                        ),
                                        duration,
                                        0,
                                        duration,
                                        file_path,
                                    )
                                )
                            except FileNotFoundError:
                                missing_audio_files += 1
                        final_results.extend(parsed_set)
    if verbose:
        logging.info(
            f"NST audio files found: {found_audio_files}\nNST audio files missing: {missing_audio_files}"
        )
        logging.info("checking for duplicates of train in test...")
    duplicates = 0
    audios_in_train = [
        x.audio_file
        for x in final_results
        if x.original_data_split == dataset_prefix + "train"
    ]
    cleaned_results = []
    for r in final_results:
        if r.original_data_split == dataset_prefix + "train":
            cleaned_results.append(r)
        else:
            if r.audio_file not in audios_in_train:
                cleaned_results.append(r)
            else:
                duplicates += 1
    if verbose:
        logging.info(
            f"{duplicates} duplicate files in test and train removed from test"
        )
    return cleaned_results


if __name__ == "__main__":

    json_path = "../json/"
    audio_path = "../audio/"

    # output = parse_nst(json_path, audio_path)
