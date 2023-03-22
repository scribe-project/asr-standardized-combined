import json
import os
from .shared_classes import consolidated_utterance


def create_sentence(tokens):
    sentence_words = []
    for token in tokens:
        if token["standardized_form"]:
            sentence_words.append(
                "{}|{}".format(token["token_text"], token["standardized_form"])
            )
        elif token["phon_ort_discrepancy"] == 1:
            sentence_words.append("{}|{}".format(token["token_text"], 1))
        else:
            sentence_words.append(token["token_text"])
    return " ".join(sentence_words)


def get_npsc_dialect(speaker_id, speaker_list):
    speaker_data = {}
    for d in speaker_list:
        if d["speaker_id"] == speaker_id:
            speaker_data = d
            break
    dialect = speaker_data["dialect"]
    if dialect == "Northern Norway":
        return "north"
    elif dialect == "Southern Norway":
        return "south"
    elif dialect == "Western Norway":
        return "west"
    elif dialect == "Trøndelag":
        return "mid"
    elif dialect == "Eastern Norway":
        return "east"
    else:
        return "unknown"


def get_npsc_gender(speaker_id, speaker_list):
    speaker_data = {}
    for d in speaker_list:
        if d["speaker_id"] == speaker_id:
            speaker_data = d
            break
    gender = speaker_data.get("gender", "unknown")
    if gender in ["male", "female", "unknown"]:
        return gender
    else:
        return "unknown"


def parse_npsc(npsc_dir):
    all_nspc_consolidated_utterance = []
    dataset_prefix = "npsc_"
    with open(
        os.path.join(npsc_dir, "project_files", "NPSC_speaker_data.json"), "r"
    ) as sf:
        speakers = json.load(sf)
    for session_name in os.listdir(npsc_dir):
        session_dir = os.path.join(npsc_dir, session_name)
        if session_name[:2] == "20" and os.path.isdir(session_dir):
            with open(
                os.path.join(session_dir, "{}_token_data.json".format(session_name)),
                "r",
            ) as open_f:
                data = json.load(open_f)
            # load the sentence data so we can get sentence starts and ends
            with open(
                os.path.join(session_dir, "{}_sentence_data.json".format(session_name)),
                "r",
            ) as open_f:
                sentence_data = json.load(open_f)
            sentence_data_by_id = {
                sd["sentence_id"]: {
                    "start_time": sd["start_time"],
                    "end_time": sd["end_time"],
                }
                for sd in sentence_data["sentences"]
            }
            audiofile = data["full_audio_file"]
            split = data["data_split"]
            for sentence in data["sentences"]:
                sent_start = sentence_data_by_id[sentence["sentence_id"]]["start_time"]
                sent_end = sentence_data_by_id[sentence["sentence_id"]]["end_time"]
                all_nspc_consolidated_utterance.append(
                    consolidated_utterance(
                        dataset_prefix + str(sentence["speaker_id"]),
                        get_npsc_gender(sentence["speaker_id"], speakers),
                        dataset_prefix + str(sentence["sentence_id"]),
                        sentence["sentence_language_code"]
                        if "sentence_language_code" in sentence
                        else sentence["tokens"][0]["language_code"],
                        create_sentence(sentence["tokens"]),
                        os.path.join(session_dir, audiofile),  # sentence["audio_file"],
                        dataset_prefix + split,
                        get_npsc_dialect(sentence["speaker_id"], speakers),
                        (sent_end - sent_start) / 1000,
                        sent_start / 1000,
                        sent_end / 1000,
                        os.path.join(session_dir, "audio", sentence["audio_file"]),
                    )
                )
    return all_nspc_consolidated_utterance


if __name__ == "__main__":
    npsc_dir = "/talebase/data/speech_raw/NPSC/"
    output = parse_npsc(npsc_dir)
