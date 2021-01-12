import argparse
from pathlib import Path
import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from utilities import dir_path
from wave_dataset import AudioWaveDataset
from utilities import LogMelExtractor
import yaml
config = yaml.safe_load(open("config.yaml"))
# yaml guide https://camel.readthedocs.io/en/latest/yamlref.html


def prepare_free_spoken_digit_dataset(path, output_path):
    """
    Given the path for the FSDD, retrieve the audio clips and force them to match the same length.
    Then, proceed to return the audio clips as waveforms as wall as their metadata
    :param path: Path for the FSDD audio files.
    :param output_path: Path for the resulting, same length, audio clips
    :return: tuple ( np.array with waveforms, DataFrame with metadata)
    """
    # For each file in the dataset modify it to fit a specified length in total frames (config)
    files = os.listdir(path)
    waveforms = []
    max_audio_length = int(config['free_spoken_digits_dataset']['duration'] * \
        config['free_spoken_digits_dataset']['sample_rate'])
    for filename in files:
        # Read File into np array
        y, sr = librosa.load(Path(path, filename), sr=config['free_spoken_digits_dataset']['sample_rate'])
        # Fit audio into array of the specified length
        y_modified_length = np.zeros(max_audio_length)
        # If clip is longer then cut it, else copy it at the begginign with padding of zeros
        if len(y) > len(y_modified_length):
            y_modified_length = y[:len(y_modified_length)]
        else:
            length_diff = len(y_modified_length) - len(y)
            starting_index = int(length_diff/2)
            y_modified_length[starting_index:(starting_index+len(y))] = y
        # Save the new audio clip
        sf.write(Path(output_path, filename), y_modified_length, sr)
        waveforms.append(y_modified_length)
    # Create the metadata df, it must have label and filename columns.
    metadata = pd.DataFrame(files, columns=['filename'])
    metadata[['label', 'speaker', 'recording_number']] = metadata.filename.str.replace('.wav', '').str.split('_', expand=True)

    # Return a list of the newly made fixed length arrays and the metadata DataFrame

    return waveforms, metadata


def save_dataset(args):
    """
    In this function we take the dataset mentioned in the parameters and tronsform it, so that at the end
    we can store the raw_waveforms, spectrograms and metadata in the specified output_path for it's usage
    later on in the project.
    :param args: Arg. Parser arguments
    :return:
    """
    dataset_path = args.dataset_path
    spectrograms = args.spectrograms
    output_path = args.output_path
    dataset = args.dataset
    batches = args.batches
    raw = args.raw
    mod_dataset_output_path = args.mod_dataset_output_path
    if dataset == 'free_spoken_digits_dataset':
        if mod_dataset_output_path is None:
            raise Exception('Make sure to set an output path for the modified version of the dataset.')

    if batches:
        raise NotImplementedError('There is no current implementation done to support batches.')

    if not batches:
        # Depending on the
        if dataset == 'synthetic_dataset':
            wave_dataset = AudioWaveDataset(path=dataset_path, name_as_metadata=True, name_separator='_')
            data = wave_dataset.audio
            metadata = wave_dataset.get_metadata()
            cols = list(metadata.columns)
            cols[-3] = 'label'
            metadata.columns = cols
        elif dataset == 'free_spoken_digits_dataset':
            data, metadata = prepare_free_spoken_digit_dataset(path=dataset_path, output_path=mod_dataset_output_path)
        else:
            data, metadata = None, None
        metadata.to_csv(Path(output_path, 'metadata.csv'))
        # If the flag for spectrograms is received, then spectrograms are created and save as a numpy array in the
        # output folder
        if spectrograms:
            spectrogram_array = []
            total_time_samples = int(config[dataset]['sample_rate'] // config['spectrogram']['hop_size'] \
                * config[dataset]['duration'])
            extractor = LogMelExtractor(config[dataset]['sample_rate'],
                                        config['spectrogram']['window_size'],
                                        config['spectrogram']['hop_size'],
                                        config['spectrogram']['n_bins'],
                                        config['spectrogram']['fmin'],
                                        config['spectrogram']['fmax'])
            for audio in data:
                mel_spectrogram = extractor.transform(audio=audio)
                # We remove the additional padding. If there is any. There shouldn't
                mel_spectrogram = mel_spectrogram[:total_time_samples].T
                # The following code is useful for visualizing the spectrograms
                # import librosa.display
                # import matplotlib.pyplot as plt
                # img = librosa.display.specshow(mel_spectrogram)
                # plt.show()
                spectrogram_array.append(mel_spectrogram)
            spectrogram_nparray = np.array(spectrogram_array)
            np.save(Path(output_path, 'spectrograms.npy'), spectrogram_nparray)
        # If the flag for raw_waveforms is received, then raw_waveforms are created and save as a numpy array in the
        # output folder
        if raw:
            raw_data_nparray = np.array(data)
            np.save(Path(output_path, 'raw_waveforms.npy'), raw_data_nparray)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a dataset of wav files stored in a specified folder')
    parser.add_argument('--spectrograms', action='store_true', help='Determines if spectrograms have to be created')
    parser.add_argument('--raw', action='store_true', help='Determines if raw files should be stored')
    parser.add_argument('--batches', action='store_true', help='Determines if batches are required for the output files'
                                                               '. This determines the output type. Numpy (not batches) '
                                                               'PyTorch dataset(batches)')
    parser.add_argument('--output_path', type=dir_path,
                        help='Specify the absolute output path to store the preprocessed dataset and the '
                             'spectrograms if requested')
    parser.add_argument('--dataset_path', type=dir_path,
                        help='Specify the absolute path (folder) containing the audio data to be preprocessed')
    parser.add_argument('--dataset', choices=['free_spoken_digits_dataset', 'synthetic_dataset', 'emotion_embeddings'],
                        default='free_spoken_digits_dataset',
                        help='Specify one of the supported datasets [free_spoken_digits_dataset, synthetic_dataset, '
                             ' emotion_embeddings]')
    parser.add_argument('--mod_dataset_output_path', type=dir_path,
                        help='Depending on the selected dataset, this parameter will be needed for defining the path'
                             ' for the resulting dataset after modifications. Datasets the need this parameter are:'
                             ' [free_spoken_digits_dataset]'
                             'spectrograms if requested')
    args = parser.parse_args()
    if not (args.spectrograms or args.raw):
        parser.error('No output specified, please select --spectrogram and/or --raw')
    print(args)
    save_dataset(args)

