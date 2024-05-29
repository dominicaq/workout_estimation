import pandas as pd
import subprocess
import os

#Filter metadata based on classes (squat, deadlifting, bench pressing)

def filter_data():
    train_annotations = pd.read_csv(r'k700-2020\annotations\train.csv')
    val_annotations = pd.read_csv(r'k700-2020\annotations\val.csv')

    classes = ['squat', 'deadlifting', 'bench pressing']

    filtered_train = train_annotations[train_annotations['label'].isin(classes)]
    filtered_val = val_annotations[val_annotations['label'].isin(classes)]

    filtered_train.to_csv('filtered_train.csv', index=False)
    filtered_val.to_csv('filtered_val.csv', index=False)


filter_data()