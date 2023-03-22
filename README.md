# Standardized combined dataset for ASR

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Repository associated to the paper *Improving Generalization of Norwegian ASR with Limited Linguistic Resources* (NoDaLiDa 2023).

This repo contains code for generating datasets on a common format from the datasets available to [SCRIBE](https://scribe-project.github.io/).
To use this functionality, you need to setup the dataset pipeline as described
[here](https://github.com/scribe-project/asr-standardized-combined/blob/main/dataset_pipeline_setup.md).

The code can be imported as a Python package or run as a CLI.

## Pip installation 

The package has been made pip installable to make ad-hoc use of the code easier. 
Pip installation is done by navigating to this directory (`asr-standardized-combined`) and running

```
pip install .
```

Once the package is installed, the code can be used in other python files e.g.

```python
from asr_standardized_combined import standardize_nbtale12
import pandas as pd

npsc_1b_bokmal = pd.read_csv('npsc_1b_bokmål.csv')
npsc_1b_bokmal = npsc_1b_bokmal.fillna('')
nbtale_1b_bokmal['standardized_text'] = standardize_nbtale12(list(nbtale_1b_bokmal['sentence_text_raw']), remove_empty=False)
```
Note that it does not yet work to run the pip-installed code as a CLI. If you intend to use the CLI functionality,
you need to run it from the root folder of the cloned repo.

## CLI functionality

See [here](https://github.com/scribe-project/asr-standardized-combined/blob/main/dataset_pipeline_setup.md#standardize-the-transcriptions)
for an explanation of the CLI functionality.

## Requirements
The code must be run on a Linux machine with ffmpeg installed. For required Python packages, see the requirements file.
