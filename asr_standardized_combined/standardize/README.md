# Standardize NPSC

### Example:
```python3 standardize_npsc.py -d /s2t_torch/datasets/NPSC_1_1 -l nn-NO -sf save_test -v```

Output:

```
INFO: Standardizing data from /s2t_torch/datasets/NPSC_1_1
INFO: Language selected: nn-NO
INFO: Standardize non-standard words: False
INFO: Substitute annotations: False
INFO: Keeping annotations (except <inaudible>) in triple-letter format but removing marking tokens
INFO: Substitute non-alphanumeric symbols outside the alphabet: True
INFO: Substitute numerals by their orthographic version: True
INFO: Remove transcriptions that only contain a single non-verbal sound: True

DEBUG: ***STANDARDIZATION OF NON-STANDARD WORDS DONE***
DEBUG: Number of sentences with non-standard words NOT standardized: 2048

DEBUG: ***REMOVING UNDERSCORES***
DEBUG: Number of unique sentences with underscored words: 22

DEBUG: ***NON-VERBAL ANNOTATIONS***
DEBUG: Number of unique sentences with annotations: 2142
DEBUG: Different annotations found in the data changed to triple-letter format and with the token removed (inaudible removed): {'<ee>': 3431, '<inaudible>': 118, '<qq>': 83, '<mm>': 16}

DEBUG: Removing utterances that only consist of one of the following: ['', ' ', 'qqq', 'eee', 'mmm']
INFO: The original list contained 8285 utterances while the standardized list contains 8280 utterances

DEBUG: Behold! See a few standardized sentences randomly selected:
     og det har vi sjekka opp på iallfall på heimesida til enova har me sjekka opp den avtalen og den seier lite om konkrete eee satsingar og du finn lite tiltak som som på ein måte er qqq liknar på det som står i anmodningsvedtaka oppmodingsvedtaka
     eee og så merkar eg meg og da at eee i budsjettdebatten her ja så har eg visst hamna i ein komité med eee ein viss handlingsretta og eee komité og med ein viss grad av utålmodigheit
     neste talar and andré n skjelstad deretter steinar reiten
     færre bønder sluttar no enn når arbeidarpartiet styrte sist
     det har vi
     og eg er glad for at ein samla komité slår fast at det som kunne vore te unnskyld slår fast det som kunne vore teke rett ut av senterpartiet sitt partiprogram nemleg at noreg sine nasjonale interesser må stå fremst i norsk tryggingspolitikk
     vi har forsøk med statleg finansiering og eldreomsorg
     så ser eg det ligg eit laust forslag på bordet eee frå venstre
     president det er jo første gong det er komme eee forslag som omhandlar eee dei marine ressursane skal inn i grunnloven
     eg veit at det er bare eitt eksempel men eg veit det finst fleir og det er for dårleg president

INFO: Saving csv to /s2t_torch/datasets/NPSC_1_1/standardized_csvs/save_test_20220616.csv
INFO: Saving config to /s2t_torch/datasets/NPSC_1_1/standardized_csvs/save_test_20220616.json
```

### Help displayed by ```python3 standardize_npsc.py -h```:

```
usage: standardize_npsc.py [-h] [-l {nb-NO,nn-NO,both}] -d DATA_DIR [-sw]
                           [-sa] [-at ANNOTATION_TOKEN]
                           [-st SUBSTITUTION_TOKEN] [-kat] [-ks] [-kn] [-ke]
                           [-sf SAVE_FILENAME] [-li] [-v]

Standardize transcriptions for ASR training and testing

optional arguments:
  -h, --help            show this help message and exit
  -l {nb-NO,nn-NO,both}, --language {nb-NO,nn-NO,both}
                        Language to select
  -d DATA_DIR, --data_dir DATA_DIR
                        Path to main directory with the raw data
  -sw, --standard_words
                        Standardize non-standard words
  -sa, --substitute_annotations
                        Substitute annotations
  -at ANNOTATION_TOKEN, --annotation_token ANNOTATION_TOKEN
                        Token used in data to mark annotations
  -st SUBSTITUTION_TOKEN, --substitution_token SUBSTITUTION_TOKEN
                        Token used to substitute annotations when -sa given
  -kat, --keep_annotation_token
                        Keep the token used to mark annotations if -sa not
                        given
  -ks, --keep_symbols   Keep not alphanumeric symbols outside the alphabet
  -kn, --keep_numerals  Keep numerals (otherwise substitute them by their
                        orthographic version)
  -ke, --keep_empty     Keep transcriptions only containing non-verbal sounds
  -sf SAVE_FILENAME, --save_filename SAVE_FILENAME
                        Saves a csv including standardized transcriptions in
                        data_dir/standardized_csvs/save_filename.csv and a
                        config json file
                        data_dir/standardized_csvs/save_filename.json with the
                        configuration chosen
  -li, --listen         Opens audio files in VSCode whose transcriptions
                        contain tokens out of alphabet after standardization
  -v, --verbose         Displays debugging information
  ```

# Standardize NBTale3

### Example:
```python3 standardize_nbtale3.py -d /s2t_torch/datasets/NB_Tale_senn3_down -v -sf GT_NBTale3```

Output:
```
INFO: Standardizing data from /s2t_torch/datasets/NB_Tale_senn3_down
INFO: Standardize non-standard words: False
INFO: Remove verbal annotations: True
INFO: Remove non-verbal annotations: True
INFO: Substitute non-alphanumeric symbols outside the alphabet: True
INFO: Substitute numerals by their orthographic version: True
INFO: Remove transcriptions that only contain annotations: True

***SENTENCE-SEGMENTED AUDIO***
INFO: Sentence-segmented audio files in /s2t_torch/datasets/NB_Tale_senn3_down/part_3_audio_segments
INFO: Total number of segments: 8649

***REMOVING EMPTY UTTERANCES***
DEBUG: Removing utterances that only consist of one or more of the following: ['', ' ']
INFO: The original list contained 8649 utterances while the standardized list contains 8286 utterances

DEBUG: Behold! See a few standardized sentences randomly selected:
     tre uker
     da er det bare stas
     og det er ikke en sak jeg liker
     og eldstesøstera mi har to barn allerede
     jeg er jo opprinnelig historiker
     først så kom den minste bukken bruse og trampa på brua
     nå har det gått en del år nå er jeg voksen
     men jeg hadde ikke noe interesse å gå videre i på pedagogikk så ville jeg fortsette med forskning
     stå opp klokka ni om morgenen spise frokost og vi gå i kirken klokka elleve
     ende

INFO: Saving csv to /s2t_torch/datasets/NB_Tale_senn3_down/standardized_csvs/GT_NBTale3_20220707.csv
INFO: Saving config to /s2t_torch/datasets/NB_Tale_senn3_down/standardized_csvs/GT_NBTale3_20220707.json
```

### Help displayed by ```python3 standardize_nbtale3.py -h```:

```
usage: standardize_nbtale3.py [-h] -d DATA_DIR [-sw] [-knva] [-kva]
                              [-at ANNOTATION_TOKEN] [-ks] [-kn] [-ke]
                              [-sf SAVE_FILENAME] [-li] [-v]

Standardize transcriptions for ASR training and testing

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        Path to main directory with the raw data
  -sw, --standard_words
                        Standardize non-standard words
  -knva, --keep_nv_annotations
                        Keep non-verbal annotations
  -kva, --keep_v_annotations
                        Keep verbal annotations
  -at ANNOTATION_TOKEN, --annotation_token ANNOTATION_TOKEN
                        Token used in data to mark annotations
  -ks, --keep_symbols   Keep not alphanumeric symbols outside the alphabet
  -kn, --keep_numerals  Keep numerals (otherwise substitute them by their
                        orthographic version)
  -ke, --keep_empty     Keep transcriptions only containing non-verbal sounds
  -sf SAVE_FILENAME, --save_filename SAVE_FILENAME
                        Saves a csv including standardized transcriptions in
                        data_dir/standardized_csvs/save_filename.csv and a
                        config json file
                        data_dir/standardized_csvs/save_filename.json with the
                        configuration chosen
  -li, --listen         Opens audio files in VSCode whose transcriptions
                        contain tokens out of alphabet after standardization
  -v, --verbose         Displays debugging information
```
