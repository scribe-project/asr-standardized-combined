from .shared_classes import (
    word_transcript,
    file_transcript,
    consolidated_utterance,
    consolidated_utterance_phon,
)
import os
import re

number_pattern = re.compile("\d+")

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
    "kor",
    "nokon",
    "gjere",
    "høgre",
    "vera",
    "vidare",
    "ligg",
    "betre",
    "ifrå",
    "handlar",
    "same",
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
    "vert",
    "særleg",
    "eigentleg",
    "fekk",
    "kunna",
    "korfor",
    "einige",
    "krevjande",
    "eigen",
    "kvart",
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
    "knytt",
    "nokre",
    "verda",
    "delar",
    "politireforma",
    "snakkar",
    "gonger",
    "takkar",
    "innan",
    "bruka",
    "atomvåpen",
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
    "såg",
    "dyrevelferda",
    "forventar",
    "verkemiddel",
    "høyring",
    "finn",
    "skriv",
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
    "trass",
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
    "utsett",
    "vegen",
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
    "krev",
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


def get_gender(speaker_id, genderdict):
    if genderdict[speaker_id] == "M":
        return "male"
    elif genderdict[speaker_id] == "K":
        return "female"
    else:
        return "unknown"


def process_trans_files_parts_1_2(trans_file):
    with open(trans_file, "r") as open_f:
        data = open_f.read()
    transcription_lines = data.split("\n")
    # remove the headers/other commented lines
    transcription_lines = [l for l in transcription_lines if len(l) > 0 and l[0] != "#"]

    # this will be a dict where keys are file names and values are file_transcript(s)
    transcription_by_file = {}
    current_file = ""
    current_transcript = file_transcript()
    current_word = ""

    for line in transcription_lines:
        if line[:5] == '"part':
            current_file = line.strip('"')
        elif len(line.split("\t")) == 4:
            word_line = line.split("\t")
            word = word_line[-1]
            if word == "<end>":
                # wrap up the file transcript
                # first wrap up the word we had on the go
                current_word.set_word_end(
                    word_line[0]
                )  # the last word ends when the <end> starts
                current_transcript.add_word(current_word)
                current_transcript.set_file_duration(word_line[1])
                transcription_by_file[current_file] = current_transcript
                current_transcript = file_transcript()
                current_word = ""
            elif word == "<start>":
                pass
            else:
                # if we have a current word, wrap it up
                if isinstance(current_word, word_transcript):
                    # first update the ending timestamp for the old word
                    current_word.set_word_end(word_line[1])
                    # then add it to the transcript
                    current_transcript.add_word(current_word)
                # now create a new transcript for the word we're currently starting
                current_word = word_transcript(word, word_line[0], word_line[1])
                # and add the first phoneme
                current_word.add_phoneme(word_line[2], word_line[0], word_line[1])
        elif len(line.split("\t")) == 3:
            # we only need to worry about updating the sound trancript
            sound_line = line.split("\t")
            # first get the sound annotation addition out of the way
            current_word.add_phoneme(sound_line[2], sound_line[0], sound_line[1])
        else:
            pass
    return transcription_by_file


def process_trans_files_parts_3(trans_file):
    with open(trans_file, "r") as open_f:
        data = open_f.read()
    transcription_lines = data.split("\n")
    # remove the headers/other commented lines
    transcription_lines = [l for l in transcription_lines if len(l) > 0 and l[0] != "#"]

    # this will be a dict where keys are file names and values are file_transcript(s)
    transcription_by_file = {}
    current_file = ""
    current_transcript = file_transcript()
    current_word = ""
    speaker_utt_counter = 1

    for line in transcription_lines:
        if line[:5] == '"part':
            # first check that we have a speaker to wrap-up
            if current_file:
                # wrap up the last speakers last utt
                # the end time should be from the last word
                found_end = False
                for word in current_transcript.get_words_transcripts():
                    if word.end != 0:
                        current_transcript.set_file_end(word.end)
                        found_end = True
                # this is the case where we have a sentence boundary at the end of a file w/ no words in the transcript
                if not found_end:
                    current_transcript.set_file_end(current_transcript.get_file_start())
                # finally actually save as normal
                utt_file_key = current_file + "_{}".format(speaker_utt_counter)
                transcription_by_file[utt_file_key] = current_transcript
            # start a new speakers
            current_file = line.strip('"')
            speaker_utt_counter = 1
            current_transcript = file_transcript()
            current_transcript.set_file_start(0)

        elif len(line.split("\t")) > 2:
            word_line = line.split("\t")
            # we'll just treat all lines as start, end, word
            start = word_line[0]
            end = word_line[1]
            word = word_line[2]
            # some sentence boundaries don't have time stamps and we'll skip those
            if word == "<sentence_boundary>" and end != "_":
                # wrap up the file transcript
                current_transcript.set_file_end(end)
                utt_file_key = current_file + "_{}".format(speaker_utt_counter)
                transcription_by_file[utt_file_key] = current_transcript
                speaker_utt_counter += 1
                current_transcript = file_transcript()
                current_transcript.set_file_start(start)
            else:
                if number_pattern.match(start) and number_pattern.match(end):
                    current_word = word_transcript(word, int(start), int(end))
                else:
                    # set the word but since we don't have word boundries we'll set them to 0
                    current_word = word_transcript(word, 0, 0)
                # then add it to the transcript
                current_transcript.add_word(current_word)
        else:
            pass
    return transcription_by_file


def get_informant_metadata(informant_data_file):
    with open(informant_data_file, "r", encoding="utf-16") as open_f:
        data = open_f.read()
    data = data.split("\n")
    informant_id_pattern = re.compile("p[1-2]_g\d{2}_[fmn][0-2]_\d(_\w)?")
    all_informants = []
    for row in data:
        row = row.split("\t")
        if informant_id_pattern.match(row[0]):
            all_informants.append(row)
    return all_informants


def get_speaker_id(spkr_id, spkr_list):
    spkr_id = os.path.basename(spkr_id)
    if spkr_id in spkr_list:
        return spkr_id
    if spkr_id[-1] in ["t", "x"]:
        # remove the _t or _x and try again
        spkr_id = spkr_id.strip("_t").strip("_x")
        return get_speaker_id(spkr_id, spkr_list)
    raise Exception(
        "Cannot find informant id: {} in know informant list".format(spkr_id)
    )


def get_nbtale_dialect(group):
    group = int(group)
    if group > 12:
        return "foreign"
    elif group < 4:
        return "north"
    elif group in [4, 5]:
        return "mid"
    elif group in [6, 7, 8, 9]:
        return "west"
    elif group == 10:
        return "south"
    else:
        return "east"


def parse_nbtale(nbtale_dir, microphone="shure", phonetic=False):
    '''By default, the file path to the Shure table microphone is given.
    For the head microphone,  microphone="sennheiser"'''
    
    dataset_prefix = "nbtale_"
    nbtale_dir = str(nbtale_dir)
    annotation_dir = os.path.join(nbtale_dir, "Annotation", "Annotation")
    informant_file = os.path.join(
        nbtale_dir,
        "Documentation/Dokumentasjon",
        "05_NB_Tale_Informantdata.txt",
    )
    informant_metadata = get_informant_metadata(informant_file)
    all_informant_ids = [x[0] for x in informant_metadata]
    informant_genders = {x[0]: x[2] for x in informant_metadata}
    all_utterances = []
    
    for annotation_file in os.listdir(annotation_dir):
        if ".trans" in annotation_file:
            part = annotation_file.split(".")[0].split("_")[1]
            if part != "3":
                # let's parse it!
                file_utterances = process_trans_files_parts_1_2(
                    os.path.join(annotation_dir, annotation_file)
                )
                for file_utt_key in file_utterances:
                    spkr_id = file_utt_key.split("-")[0]
                    utt_id = file_utt_key.split("/")[-1]
                    spkr_id = get_speaker_id(spkr_id, all_informant_ids)
                    group = spkr_id.split("_")[1][1:]
                    file_utt = file_utterances[file_utt_key]
                    if not phonetic:
                        all_utterances.append(
                            consolidated_utterance(
                                dataset_prefix + spkr_id,
                                get_gender(spkr_id, informant_genders),
                                dataset_prefix + utt_id,
                                pred_nynorsk(
                                    file_utt.get_orthographic_readable(), words_nn
                                ),
                                file_utt.get_orthographic_readable(),
                                os.path.join(
                                    nbtale_dir,
                                    f"{microphone}_{part}",
                                    file_utt_key + ".wav",
                                ),
                                f"nb_tale_part_{part}",
                                get_nbtale_dialect(group),
                                file_utt.get_file_duration() / 1000,  # convert to s
                                0,
                                file_utt.get_file_duration() / 1000,  # convert to s
                                os.path.join(
                                    nbtale_dir,
                                    f"{microphone}_{part}",
                                    file_utt_key + ".wav",
                                ),
                            )
                        )
                    else:
                        utt_words = [
                            {
                                "word": x.label,
                                "start": float(x.start) / 1000,
                                "end": float(x.end) / 1000,
                            }
                            for x in file_utt.get_words_transcripts()
                        ]
                        utt_phones = [
                            {
                                "phone": x.label,
                                "start": float(x.start) / 1000,
                                "end": float(x.end) / 1000,
                            }
                            for x in file_utt.get_all_phonemes()
                        ]
                        all_utterances.append(
                            consolidated_utterance_phon(
                                dataset_prefix + spkr_id,
                                get_gender(spkr_id, informant_genders),
                                dataset_prefix + utt_id,
                                pred_nynorsk(
                                    file_utt.get_orthographic_readable(), words_nn
                                ),
                                file_utt.get_orthographic_readable(),
                                utt_words,
                                utt_phones,
                                os.path.join(
                                    nbtale_dir,
                                    f"{microphone}_{part}",
                                    file_utt_key + ".wav",
                                ),
                                f"nb_tale_part_{part}",
                                get_nbtale_dialect(group),
                                file_utt.get_file_duration() / 1000,  # convert to s
                                0,
                                file_utt.get_file_duration() / 1000,  # convert to s
                            )
                        )
            else:
                if phonetic:
                    pass
                else:
                    file_utterances = process_trans_files_parts_3(
                        os.path.join(annotation_dir, annotation_file)
                    )
                    final_numbers = re.compile("_\d+($|.wav)")
                    for file_utt_key in file_utterances:
                        spkr_id = file_utt_key.split("-")[0]
                        utt_id = file_utt_key.split("/")[-1]
                        spkr_id = get_speaker_id(spkr_id, all_informant_ids)
                        group = spkr_id.split("_")[1][1:]
                        file_utt = file_utterances[file_utt_key]
                        file_utt_key_denumbered = final_numbers.sub("", file_utt_key)
                        all_utterances.append(
                            consolidated_utterance(
                                dataset_prefix + spkr_id,
                                get_gender(spkr_id, informant_genders),
                                dataset_prefix + utt_id,
                                "nb-NO",
                                file_utt.get_orthographic_readable(),
                                os.path.join(
                                    nbtale_dir,
                                    f"{microphone}_{part}",
                                    file_utt_key_denumbered + ".wav",
                                ),
                                f"nb_tale_part_{part}",
                                get_nbtale_dialect(group),
                                file_utt.get_file_duration() / 1000,  # convert to s
                                file_utt.get_file_start() / 1000,
                                file_utt.get_file_end() / 1000,  # convert to s
                            )
                        )

    return all_utterances


if __name__ == "__main__":
    trans_file_path = "/talebase/data/speech_raw/NBTale"
    transcription = parse_nbtale(trans_file_path)
