import logging 
import os
from pydub import AudioSegment  # to segment the audio
from subprocess import call  # for opening audios in VSCode

logger = logging.getLogger(__name__)

def export_audio_segments(df, filename, audio_path_total): 
    """
    Given the path to downloaded data and a dataframe with the audio file names and start and end times,
    creates the path to the sentence-segmented audio files and the files themselves from the full audio.
    The sentence-segmented files are created in wav format, PCM 16bits and 16kHz sample rate.
    NOTE: it's trivial to let the user choose these parameters as arguments and parse them. This is
    unnecessary now but could be considered in future implementations.

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
    Nothing, it executes the actions described above.
    """
    
    start = round(df.start_time * 1000)
    end = round(df.end_time * 1000)
    rel_path_to_filename = os.path.join("/", *filename.split("/")[:-1])
    if not os.path.exists(rel_path_to_filename):
        os.makedirs(rel_path_to_filename)
    if not os.path.isfile(filename):
        audioseg = AudioSegment.from_file(audio_path_total)
        intaudio = audioseg[start:end]
        intaudio = intaudio.set_frame_rate(16000)
        intaudio.export(filename, format="wav", bitrate="16k")

def out_of_alphabet(transcription_list, alphabet):
    """
    Given a list of transcriptions (strings) and an alphabet (list of strings)
    identifies characters, words, sentences, and sentence indices that contain 
    characters outside the given alphabet.
    
    Parameters
    ----------
    transcription_list: list of strings
        List of transcriptions that we wish to analyze
    alphabet: list of strings
        List of characters that conforms the allowed alphabet (default is
        'a b c d e f g h i j k l m n o p q r s t u v w x y z å ø æ - é –'.split())
        
    Returns
    -------
    Tuple with four lists of equal length:

    del_letters: list of strings
        List of characters outside the given alphabet
    del_words: list of strings
        List of words (tokens separated by a space) that contain one or more characters
        outside the given alphabet
    del_sentences: list of strings
        Subset of sentences from transcription_list that contain one or more characters
        outside the given alphabet
    index_sentence: list of integers
        List of integers indicating the index positions from transcription_list that
        contain one or more characters outside the given alphabet
    """
    del_letters = []
    del_words = []
    index_sentence = []
    del_sentences = []
    
    for (i,x) in enumerate(transcription_list):
        for y in x.lower().split(): # this makes that only transcriptions in lower case work
            chars = list(y)
            for letter in chars:
                if letter not in alphabet:
                    if letter not in del_letters:
                        del_letters.append(letter)
                    if y not in del_words:
                        del_words.append(y)
                    if i not in index_sentence:
                        index_sentence.append(i)
                        del_sentences.append(x)
    
    return del_letters, del_words, del_sentences, index_sentence

def play_audios(
    standardized_transcripts, out_of_alphabet_info, audio_list
):
    """
    For debugging purposes and creating the correct substitution rules, this function opens the audio files
    whose transcripts contain tokens out of alphabet.
    NOTE: The instruction for opening the audio file is particular for VSCode.

    Parameters
    ----------
    standardized_transcripts: list of strings
        List of transcripts that we wish to analyze for out of alphabet tokens and listen to their pronunciation
    out_of_alphabet_info: 4-tuple of lists
        This is the output of out_of_alphabet(standardized_transcripts), computed beforehand for efficiency
    audio_list: list of strings
        List of sentence-segmented audio files

    Returns
    -------
    As for now, nothing is returned. It requires your input so that not all the audio files are opened at once.
    Can return a dictionary {audio_file: modified_transcription} with little modification of the code below
    """
    # Sentence replacement given an input after playing audio file (requires personalizing to the structure of the data)
    # Commented code below is work in progress for creating a dictionary of substitutions "on the fly"
    # sentence_Dict = {}
    for (i, s) in enumerate(standardized_transcripts):
        if i in out_of_alphabet_info[3]:
            logger.info("Audio: {}".format(audio_list[i]))
            logger.info("Transcript: {}".format(standardized_transcripts[i]))
            call(["code", audio_list[i]])
            input("Press enter to open the next audio")
    #        new_sentence = input('Enter the correct sentence:')
    #        standardized_transcripts[i] = new_sentence
    #        print('After: {}'.format(standardized_transcripts[i]))
    #        sentence_Dict[audio_list[i]] = standardized_transcripts[i]
    #        print(sentence_Dict)

def play_checks(check_list, trans_list, audio_list):
    """
    Function for exploration purposes. It allows to listen to the audio files whose transcriptions
    contain the strings given in check_list. This is to check for pronunciation and possible
    transcription errors, and use that knowledge to create the substitution dictionaries.
    NOTE: the instruction to reproduce the audio files is specific for VSCode.

    Parameters
    ----------
    check_list: list of strings
        List of strings that we want to find in the transcriptions and play their corresponding
        sentence-segmented audio file.
    trans_list: list of strings
        List of transcriptions.
    audio_list: list of strings
        List of sentence-segmented audio files.

    Returns
    -------
    Nothing, it executes the actions described above.
    """
    for (t, a) in zip(trans_list, audio_list):
        for i in check_list:
            if i in t:
                logger.info("Audio: {}".format(a))
                logger.info("Transcript: {}".format(t))
                logger.info("Opening audio file")
                call(["code", a])
                input("Press enter to open the next audio")

def remove_empty_utt(transcription_list, audio_list, annotation_token, alphabet):
    """
    Removes utterances that are empty or just contain annotations for acoustic events (given in wordDic_annot
    and wordDic_sounds above). It can be easily modified to remove utterances that contain only *one* annotation.
    NOTE: It assumes that these annotations are all marked with the same token, which in the case of NBTale3 is '<'.

    Parameters
    ----------
    transcription_list: list of strings
        List of transcribed sentences.
    audio_list: list of strings
        List of sentence-segmented audio files.
    annotation_token: str
        Token used to mark annotations of acoustic events.
    remove_empty: bool
        Removes utterances (True) or not (False).

    Returns
    -------
    standardized_audios, standardized_sentences: tuple of lists of strings
        Lists of sentence-segmented audio files and their corresponding transcriptions, where the empty ones
        have been removed (or not) according to the description above.
    """
    standardized_audios = audio_list
    _, _, del_sentences, _ = out_of_alphabet(transcription_list, alphabet=alphabet)
    hesitation_words = [
        word
        for sentence in del_sentences
        for word in sentence.lower().split()
        if annotation_token in word
    ]
    filters = ["", " "] + list(set(hesitation_words))
    logger.info("")
    logger.info("***REMOVING EMPTY UTTERANCES***")
    # logging.debug('Removing utterances that only consist of one of the following: {}'.format(filters))
    # standardized_audios, standardized_sentences = zip(*[(a, s) for (a, s) in zip(standardized_audios, transcription_list) if s not in filters])
    logger.debug(
        "Removing utterances that only consist of one or more of the following: {}".format(
            filters
        )
    )
    standardized_audios, standardized_sentences = zip(
        *[
            (a, s)
            for (a, s) in zip(standardized_audios, transcription_list)
            if not all(w in filters for w in s.split())
        ]
    )
    logger.info(
        "The original list contained {} utterances while the standardized list contains {} utterances".format(
            len(transcription_list), len(standardized_sentences)
        )
    )
    return standardized_audios, standardized_sentences

def replace_symbols(sentence, wordDic_sym):
    """
    For a sentence, it replaces a set of expressions containing tokens out of the alphabet
    by its expression in terms of characters in the alphabet.
    NOTE: the dictionary wordDic_sym is predefined after exploration of the data. We recommend defining
    two dictionaries: one for numerals, one for tokens with other symbols; and apply for each dictionary.
    New corpora require finding these expressions and redefining the replacement dictionary. This can be 
    done by setting substitute_symbols=False and verbose=True in standardize and the script will output 
    expressions with characters out of alphabet.

    Parameters
    ----------
    sentence: str
    wordDic_sym: dict
        Predefined dictionary with substitutions.
    
    Returns
    -------
    sentence: str
        Sentence where the expressions with tokens in wordDic_sym (keys) have been substituted
        by the values.
    """
    for ini, out in wordDic_sym.items():
        sentence = sentence.replace(ini, out)
    return sentence 

def substitute_hesitations(
    sentence, keep_annotations, annotation_token, substitution_token, wordDic_hes={}
):
    """
    For a sentence, it finds the non-verbal annotations identified by a given annotation_token
    and replaces them (or not) by a substitution_token.

    Parameters
    ----------
    sentence: str
    keep_annotations: bool
        Determines whether we want to replace annotations by a token
    annotation_token: str
        The symbol used in the corpus that identifies non-verbal annotations.
        Must be unique (default is '<' for the NPSC)
    substitution_token: str
        If supplied, this is token used to substitute all non-verbal annotations. Else (if False) then keep_annotation logic applies
    

    Returns
    -------
    sentence: str
        Sentence where the non-verbal annotations have been substituted or not
    """

    # behaviour where we delete all tokens can been achieved by setting substitution_token = " "
    # keep_annotations == False will do the wordDict_hes subs and then remove any remaining tokens
    # keep_annotations == True will do...nothing

    if substitution_token:
        return " ".join([(substitution_token if annotation_token in word else word) for word in sentence.split()])
    elif not keep_annotations:
        replace_sentence = replace_symbols(sentence, wordDic_hes)
        return " ".join([word for word in replace_sentence.split() if annotation_token not in word])
    else:
        return sentence

def substitute_underscores(sentence):
    """
    For a sentence, replaces the underscore '_' for a space. 
    This is needed as underscores appear in the standardized version of some tokens
    when they are composed of multiple words, e.g. i_forhold_til.
    It is important to apply this function before any other tokens
    (such as non-verbal annotations) are substituted by an underscore.
    """
    return sentence.replace('_', ' ')