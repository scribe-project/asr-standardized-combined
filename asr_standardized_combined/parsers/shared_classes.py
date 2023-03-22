from dataclasses import dataclass
import re
from typing import List


@dataclass
class consolidated_utterance:
    speaker_id: str
    speaker_gender: str
    sentence_id: str
    sentence_language_code: str
    sentence_text_raw: str
    audio_file: str
    original_data_split: str
    dialect: str
    sentence_duration_s: float
    start_time: int = 0
    end_time: int = 0
    segmented_audio_file: str = ""


@dataclass
class consolidated_utterance_phon:
    speaker_id: str
    speaker_gender: str
    sentence_id: str
    sentence_language_code: str
    sentence_text_raw: str
    wordslist_raw: List[dict]
    phonelist_raw: List[dict]
    segmented_audio_file: str
    original_data_split: str
    dialect: str
    sentence_duration_s: float
    start_time: int = 0
    end_time: int = 0


@dataclass
class transcript_annotation:
    start: float
    end: float
    label: str
    score: float = 0


class word_transcript:
    def add_phoneme(self, phoneme: str, start: int, end: int) -> None:
        self.phonemes.append(transcript_annotation(start, end, phoneme))

    def add_phoneme_annotation(self, phoneme_annotation: transcript_annotation) -> None:
        self.phonemes.append(phoneme_annotation)

    def set_word_start(self, start: int) -> None:
        self.word_start = start

    def set_word_end(self, end: int) -> None:
        self.word_end = end

    def get_phonemes(self) -> list:
        return self.phonemes

    def get_word(self) -> str:
        return self.word

    def get_word_annotation(self) -> transcript_annotation:
        return transcript_annotation(self.word_start, self.word_end, self.word)

    def __init__(self, word: str, start=0, end=0) -> None:
        self.word = word
        self.word_start = start
        self.word_end = end
        self.phonemes = []


class file_transcript:
    def add_word(self, word_annotation: word_transcript):
        self.word_table[word_annotation.get_word()] = word_annotation
        self.sentence.append(word_annotation.get_word())

    def get_all_phonemes(self):
        all_phonemes = []
        for w in self.sentence:
            all_phonemes.extend(self.word_table[w].get_phonemes())
        return all_phonemes

    def get_word(self, word):
        if word in self.word_table:
            return self.word_table[word]
        return None

    def get_words_transcripts(self):
        words_transcript = []
        for w in self.sentence:
            words_transcript.append(self.word_table[w].get_word_annotation())
        return words_transcript

    def get_word_list(self):
        return self.sentence

    def get_word_count(self):
        return len(self.sentence)

    def print_orthographic_readable(self):
        print(" ".join(self.sentence))

    def get_orthographic_readable(self):
        return " ".join(self.sentence)

    def set_file_duration(self, duration):
        self.file_duration = float(duration)

    def get_file_duration(self):
        if self.file_duration:
            return self.file_duration
        else:
            return self.file_end - self.file_start

    def set_file_start(self, start):
        self.file_start = float(start)

    def get_file_start(self):
        return self.file_start

    def set_file_end(self, end):
        self.file_end = float(end)

    def get_file_end(self):
        return self.file_end

    def __init__(self):
        # this will maintain the word order
        self.sentence = []
        # this will make looking up word timestamps and phones easier
        self.word_table = {}
        self.file_duration = 0
        self.file_start = 0
        self.file_end = 0
