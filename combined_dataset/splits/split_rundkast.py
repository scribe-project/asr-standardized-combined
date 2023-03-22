import pandas as pd
import numpy as np
import os
import argparse
import logging

# Parser
parser = argparse.ArgumentParser(
    description="Split dataset in train/test/eval close to a 80/10/10 proportion"
)

parser.add_argument(
    "-d",
    "--csv_dir",
    type=str,
    required=True,
    help="Path to csv with Rundkast data",
)

logger = logging.getLogger(__name__)

# Auxiliary function
def speaker_overlap(list1, list2):
    c = 0
    for e1 in list1:
        if e1 in list2:
            c+=1
    return c

def split_data(csv_dir):
    df = pd.read_csv(csv_dir, header=None)
    d = df.groupby(5)[0].apply(lambda x: len(list(np.unique(x)))).to_dict()
    sorted_programs_by_N_speakers = sorted(d, key=d.get, reverse=True)
    test_programs = sorted_programs_by_N_speakers[:11] # maximum variability in terms of speakers
    eval_programs = sorted_programs_by_N_speakers[11:22]
    train_programs = sorted_programs_by_N_speakers[22:]

    # Checks
    for p in test_programs:
        if p in eval_programs: print(p, 'in eval')
        if p in train_programs: print(p, 'in train')
    for p in eval_programs:
        if p in train_programs: print(p, 'in train')

    df_train = df[df[5].isin(train_programs)]
    df_eval = df[df[5].isin(eval_programs)]
    df_test = df[df[5].isin(test_programs)]

    for df, name, i in zip([df_train, df_eval, df_test],['Train', 'Eval', 'Test'], [0,1,2]):
        print('*** {} ***'.format(name))
        print(df.shape[0], 'utterances')
        total_time = df[8].sum()
        print('Duration in hours: {}'.format(round(1/3600*total_time, 2)))
        list_speakers = df[0].unique()
        print('Number of different speakers: {}'.format(len(list_speakers)))
        print('Gender (in terms of number of utterances):')
        for line in df[1].value_counts(dropna=False, normalize=True).items():
            print(' {}: {}%'.format(line[0], round(100*line[1], 2)))
        print('Gender (in terms of time speaking):')
        for line in df.groupby(1)[8].sum().items():
            print(' {}: {}%'.format(line[0], round(100*line[1]/total_time, 2)))
        print('Written language:')
        for line in df[3].value_counts(dropna=False, normalize=True).items():
            print(' {}: {}%'.format(line[0], round(100*line[1], 2)))
        print('Dialect group:')
        for line in df[7].value_counts(dropna=False, normalize=True).items():
            print(' {}: {}%'.format(line[0], round(100*line[1], 2)))
        # Creating split dir and saving csv
        split_dir = os.path.join(*csv_dir.split('/')[:-1], name.lower())
        print('Creating split directory {}'.format(split_dir))
        os.mkdir(split_dir)
        head_filename = csv_dir.split('/')[-1].split('.')[0]
        print("Saving csv {}_{}.csv in {}".format(head_filename, name.lower(), split_dir))
        df.to_csv("{}/{}_{}.csv".format(split_dir, head_filename, name.lower()), header=False, index=False)
        print()
        
    # If you find this line I owe you a beer
    speakers_train = df_train[0].unique()
    speakers_eval = df_eval[0].unique()
    speakers_test = df_test[0].unique()

    print('*** Overlaps in speakers ***')
    print('Train-Eval: {} speakers in common'.format(speaker_overlap(speakers_train,
                                                                    speakers_eval)))
    print('Train-Test: {} speakers in common'.format(speaker_overlap(speakers_train,
                                                                    speakers_test)))
    print('Test-Eval: {} speakers in common'.format(speaker_overlap(speakers_test,
                                                                    speakers_eval)))


if __name__ == "__main__":

    args = parser.parse_args()

    # Options chosen
    logger.info("Splitting data from {}".format(args.csv_dir))

    split_data(args.csv_dir)
