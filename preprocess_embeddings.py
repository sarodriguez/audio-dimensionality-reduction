import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from utilities import dir_path


def save_embedding(args):
    """
    Following the parameters specified in the argument parser, preprocess the received embeddings and store them along
    their metadata.
    :param args: Argument parser args
    :return:
    """
    output_path = args.output_path
    embedding_path = args.embedding_path
    with open(embedding_path, 'rb') as f:
        emb = pickle.load(f)

    embeddings, _, metadata_list = emb
    metadata = pd.DataFrame(metadata_list, columns=['filename'])
    filename_columns = ['family_name', 'gender', 'year', 'segment', 'sentiment', 'emotion', 'depression']
    metadata[filename_columns] = metadata.filename.str.replace('.wav', '').str.split('_', expand=True)
    metadata = metadata.replace({
        'gender': {
            'M1': 'Female',
            'P1': 'Male'
        },
        'sentiment': {
            's1': 'Negative',
            's2': 'Neutral',
            's3': 'Positive'
        },
        'emotion': {
            'e0': 'Anger',
            'e1': 'Fear',
            'e2': 'Joy',
            'e3': 'Sad',
            'e4': 'Neutral'
        },
        'depression': {"D{}".format(i): i for i in range(22)}
    })
    # Establish the label column to be used by the dashboard.
    metadata['label'] = metadata.family_name.str[:8]
    # metadata['label'] = metadata['emotion']
    # metadata['label'] = metadata['sentiment']
    metadata.to_csv(Path(output_path, 'metadata.csv'))
    embeddings = np.array(embeddings)
    np.save(Path(output_path, 'embeddings.npy'), embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a pickle file including embeddings and file names')
    parser.add_argument('--output_path', type=dir_path,
                        help='Specify the absolute output path to store the embedding matrix')
    parser.add_argument('--embedding_path',
                        help='Specify the absolute path for the embeddings (including filename and extension)')

    args = parser.parse_args()
    save_embedding(args)
