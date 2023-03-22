# Setting up the dataset pipeline

## Introduction
This is a description of how to set up the dataset pipeline. Once the dataset pipeline is set up on a system, you can generate datasets with a common format from the available speech corpora. For now, the pipeline only gives access to orthographic transcriptions, but we plan to add phonemic transcriptions in the future. This code is tested on Linux.

## Fetching the raw data
All of our datasets except for Rundkast come from Språkbanken. The corpora should be unpacked to the subdirectories `data/nst`, `data/storting` and `data/nbtale`. When you, at a later stage, generate datasets from these corpora, the dataset csv files will be found in these same subdirectories too. 

## Installing the combined dataset tool
We developed a tool for parsing the different corpora and producing standardized datasets. The code can be used both as a Python library and as a CLI. See [the Github repository](https://github.com/scribe-project/asr-standardized-combined) for how to use it as a Python library and how to pip-install the code.

## Standardizing the audio files
All corpora except the NPSC and NB Tale have mono audio files with a sample rate of 16kHz. In order to make a common dataset of all the corpora, the audio files of the NPSC and NB tale should be downsampled and converted to mono. We advice to use FFMPEG. If you want, you can make a copy of the directories with the original data first.

`
1. Change directory to the root directory of the combined dataset tool
`cd /path/to/combined_dataset/data/`
2. Run audio standardization on the NPSC data
```
cd storting
myvar=$(find . -name '*.wav')
for f in $myvar; do ffmpeg -i $f -ar 16000 -ac 1 "$f.wav" && mv -f "$f.wav" "$f"; done
```
3. Run audio standardization on the NB Tale data
```
cd ../nbtale
myvar=$(find . -name '*.wav')
for f in $myvar; do ffmpeg -i $f -ar 16000 -ac 1 "$f.wav" && mv -f "$f.wav" "$f"; done
```

## Standardize the transcriptions
With all this in place, you can generate standardized CSV files from all the corpora.
1. Change directory to the root directory of the combined dataset tool
`cd /path/to/combined_dataset/`
2. Generate transcription CSVs for the different datasets. The commands below are examples. To see what command line arguments are available to you, you can use the argument `--help`. `-d` (or `--data_dir`) is the data directory and `-sf` (or `--save_filename`) is the file name of the CSV and the metadat JSON files (see below).
```
python -m asr-standardized-combined.standardize.standardize_nbtale12 -d /path/to/storage/directory/nbtale -sf nbtale12 # NB Tale module 1 and 2
python -m asr-standardized-combined.standardize.standardize_nbtale3 -d /path/to/storage/directory/nbtale -sf nbtale3 # NB Tale module 3
python -m asr-standardized-combined.standardize.standardize_npsc -d /path/to/storage/directory/storting -sf npsc
python -m asr-standardized-combined.standardize.standardize_nst -d /path/to/storage/directory/nst -sf nst
```
Note that some of these scripts may take some time to run. The first time `asr-standardized-combined.standardize.standardize_nbtale3` is run, utterance-segmented audio files are produced, which are stored in `/path/to/storage/directory/nbtale/part_3_audio_segments`. The other corpora already have segmented audio files.

When you run a standardization script, a CSV with the file name you have given and a date stamp is produced in the subdirectory `standardized_csvs/` of the corpus directory. A similarly named JSON file is also produced, with the configuration of the particular run.

## Description of the CSV file
The transcription CSV file has 13 columns:
1. speaker id
2. speaker gender
3. segment id (Note that this id is derived from the original dataset and may not be unique)
4. language code (`nb-NO`, `nn-NO` or `other`)
5. non-standardized (raw) transcription of the segment
6. path to the non-segmented audio file
7. name of the original data split
8. dialect region (`east`, `west`, `south`, `mid`, `north` or `unknown`)
9. duration of the segment
10. start time in seconds from the start of the non-segmented audio file
11. end time in seconds from the start of the non-segmented audio file
12. path to the segmented audio file
13. standardized transcription of the segment. The parameters of the standardization can be controlled from the command line. Use the `--help` flag to see the possibilies available to you. 

## Generating data splits
From the CSVs of NST and NPSC datasets, you can generate smaller csvs with the canonical datasplits. The splits will be found in subdirectories `Train`, `Eval` and `Test` under `standardized_csvs` (or under whichever folder the full dataset is located in). 
```
python -m combined_dataset.splits.split_npsc -d /path/to/storage/directory/storting/standardized_csvs/name_of_file.csv
python -m combined_dataset.splits.split_nst -d /path/to/storage/directory/nst/standardized_csvs/name_of_file.csv
```

## Example code for loading a dataset in Pandas

```
import pandas as pd

cols = [
    "speaker_id",
    "gender",
    "utterance_id",
    "language",
    "raw_text",
    "full_audio_file",
    "original_data_split",
    "region",
    "duration",
    "start",
    "end",
    "utterance_audio_file",
    "standardized_text",
]

nbtale12 = pd.read_csv("nbtale12_20221003.csv", names=cols)
```
