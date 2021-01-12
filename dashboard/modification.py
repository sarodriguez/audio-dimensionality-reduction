import librosa
from pathlib import Path
from utilities import LogMelExtractor
import soundfile as sf
import numpy as np
import os
import base64

class Modifications:
    """
    Class that handles all the supported modifications applied over audio clips.
    """
    def __init__(self):
        self.id_cut = 1
        self.id_modify_amplitude = 2
        self.modifications = {
            self.id_cut: ('Cut Clip', 'Cut section from audio clip'),
            self.id_modify_amplitude: ('Modify Amplitude', 'Modify wave amplitude')
        }

    def get_ids(self):
        return list(self.modifications.keys())

    def get_name_by_id(self, key):
        return self.modifications[key][0]

    def get_description_by_id(self, key):
        return self.modifications[key][1]

    def apply_modifications(self, selected_index, modifications, metadata, dataset_path):
        """
        Apply the modifications select in the user interface given the received parameters
        :param selected_index: Index for the Audio file to modify
        :param modifications: Dictionary containing all the Modifications to be done, coming from Dash UI
        :param metadata: DataFrame with the metadata for the existing audio files, main columns for as are
            label and filename. We should append a new row to the metadata co
        :param dataset_path: Path containing the dataset
        :return:
        """
        # First we need to parse the modifications, since we receive a dictionary generated by Dash
        parsed_modifications = self.parse_modifications(modifications)
        # After having a list with the parsed modifications we need to apply them, if they are valid, if they're not
        # we just ignore them. We could add error messages but it will definitely be time consuming
        audio_path = Path(dataset_path, metadata.filename[selected_index])
        raw_waveforms, sample_rate = librosa.load(audio_path, sr=None)
        raw_waveforms = self.apply_parsed_modifications(parsed_modifications, raw_waveforms, sample_rate)
        return raw_waveforms, sample_rate

    def submit_modifications(self, selected_index, modifications, results, models, metadata, dataset_path, config, dataset):
        """
        Following the parameters, apply a modification and submit it into arrays and DataFrames containing the information
        displayed in the application
        :param selected_index: Index for the Audio file to modify
        :param modifications: Dictionary containing all the Modifications to be done, coming from Dash UI
        :param results: dictionary containing all the current 2d representations of audio clips. Here we should append
            the new modified audio clip
        :param models: dictionary containing all the models
        :param metadata: DataFrame with the metadata for the existing audio files, main columns for as are
            label and filename. We should append a new row to the metadata co
        :param dataset_path: Path containing the dataset
        :param config: Global Configuration
        :param dataset: Dataset name
        :return:
        """
        # First we apply the modifications
        raw_waveforms, sample_rate = self.apply_modifications(selected_index, modifications, metadata, dataset_path)
        return self.predict_and_add_new_audio(config, raw_waveforms, models, results, metadata, dataset_path, dataset,
                                              selected_index, sample_rate)

    def parse_modifications(self, modifications):
        parsed_modifications = []
        for mod in modifications:
            modification_description = mod['props']['children'][0]['props']['children']
            # If the modification is Cut
            if modification_description == self.modifications[self.id_cut][1]:
                start_time = mod['props']['children'][1]['props']['children'][1]['props']['value']
                end_time = mod['props']['children'][2]['props']['children'][1]['props']['value']
                # We create a tuple with the modification id (Cut) and the start and end value
                cut_tuple = (self.id_cut, start_time, end_time)
                parsed_modifications.append(cut_tuple)
            # If the modification is Modify Amplitude
            elif modification_description == self.modifications[self.id_modify_amplitude][1]:
                multiplier = mod['props']['children'][1]['props']['children'][1]['props']['value']
                start_time = mod['props']['children'][2]['props']['children'][1]['props']['value']
                end_time = mod['props']['children'][3]['props']['children'][1]['props']['value']
                modify_amplitude_tuple = (self.id_modify_amplitude, multiplier, start_time, end_time)
                parsed_modifications.append(modify_amplitude_tuple)
            else:
                pass
        return parsed_modifications

    def apply_parsed_modifications(self, parsed_modifications, raw_waveforms, sample_rate):
        for mod in parsed_modifications:
            if mod[0] == self.id_cut:
                _, start_time, end_time = mod
                start_index, end_index = int(start_time * sample_rate), int(end_time * sample_rate)
                raw_waveforms[start_index:end_index] = 0
            if mod[0] == self.id_modify_amplitude:
                _, multiplier, start_time, end_time = mod
                start_index, end_index = int(start_time * sample_rate), int(end_time * sample_rate)
                raw_waveforms[start_index:end_index] = raw_waveforms[start_index:end_index] * multiplier
        return raw_waveforms

    def clear_modifications(self, results, metadata, dataset_path):
        """
        Clears all the modifications and uploaded clips from the results, form the metadata, and removes their audio
        files from the hard drives.
        :param results: dictionary containing all the current 2d representations of audio clips. Here we should remove
        the results for the edited/added clips.
        :param metadata: DataFrame with the metadata. Here we should remove the metadata of the edited/added clips
        :param dataset_path: the path containing the audio files. This is the path for the files that will be deleted
        by this function
        :return:
        """
        # This function takes the metadata and results and removes the edited/generated samples.
        # In other words, labels with the suffix '_edited' wil be removed from the metadata and from all the results.
        edited_mask = metadata.filename.str.endswith('_edited.wav') | metadata.filename.str.endswith('_uploaded.wav')
        filenames = list(metadata.loc[edited_mask, 'filename'])
        for filename in filenames:
            os.remove(Path(dataset_path, filename))
        metadata.drop(metadata.index[edited_mask], inplace=True)
        remaining_samples = len(metadata)
        for data_type in results:
            # for model_name, result in results[data_type].items():
            #     result = result[:remaining_samples, :]
            for model_name in results[data_type]:
                results[data_type][model_name] = results[data_type][model_name][:remaining_samples, :]

    def predict_and_add_new_audio(self, config, raw_waveforms, models, results, metadata, dataset_path, dataset,
                                  selected_index=None, sample_rate=None, filename=None):
        """
        Following the parameters, creates the low-dimensional representation of an audio file for all of
        the models. This representation is added to the results and the metadata. The function modifies the
        objects received as a parameter.
        :param config: The Global configuration object.
        :param raw_waveforms: The raw waveforms to be used for creating the lower dimension representation using the models

        :param models: dictionary containing all the models
        :param results: dictionary containing all the current 2d representations of audio clips. Here we should append
            the new modified audio clip representations.
        :param metadata: Dataframe with the metadata. Here we should add the metadata for the new audio
        :param dataset_path: the path for storing the new audio file.
        :param dataset: Dataset name
        :param selected_index: The selected index, if editing a clip. Can be none when uploading a clip
        :param sample_rate: The sample rate, can be none since this comes from the confgi as well
        :param filename: The filename of the audio clip, if uploading. Can be none when editing a clip.
        :return: DataFrame metadata with the new audio metadata.
        """
        extractor = LogMelExtractor(config[dataset]['sample_rate'],
                                    config['spectrogram']['window_size'],
                                    config['spectrogram']['hop_size'],
                                    config['spectrogram']['n_bins'],
                                    config['spectrogram']['fmin'],
                                    config['spectrogram']['fmax'])
        raw_waveforms_2d = raw_waveforms.reshape(1, -1)
        mel_spectrogram = extractor.transform(audio=raw_waveforms)
        # We remove the additional padding. If there is any. There shouldn't
        total_time_samples = int(config[dataset]['sample_rate'] // config['spectrogram']['hop_size'] \
                             * config[dataset]['duration'])
        mel_spectrogram = mel_spectrogram[:total_time_samples].T
        spectrogram_flattened = mel_spectrogram.reshape(1, -1)
        # When we have the modifications we proceed to create the 2d representation in each of the results array
        # for each model. We also add a row to the metadata including the newest information for the edited clip.
        for data_type in models:
            for model_name, model in models[data_type].items():
                if model_name == config['models']['param_umap']:
                    if data_type == 'spectrograms':
                        modified_2d_representation = model.predict(mel_spectrogram.reshape(1, mel_spectrogram.shape[0],
                                                                                           mel_spectrogram.shape[1], 1))
                    else:  # data_type == raw_waveforms
                        modified_2d_representation = model.predict(raw_waveforms_2d.reshape(raw_waveforms_2d.shape[0],
                                                                                            raw_waveforms_2d.shape[1],
                                                                                            1,
                                                                                            1))
                else:
                    if data_type == 'spectrograms':
                        modified_2d_representation = model.transform(spectrogram_flattened)
                    else:  # data_type == raw_waveforms
                        modified_2d_representation = model.transform(raw_waveforms_2d)
                results[data_type][model_name] = np.append(results[data_type][model_name], modified_2d_representation,
                                                           axis=0)
        if selected_index is not None:
            unedited_metadata = metadata.loc[selected_index, :]
            edited_metadata = unedited_metadata.copy()
            # The label will now have an '_edited' suffix, so users can easily differentiate the edited clips from the
            # original ones
            edited_metadata['label'] = edited_metadata['label'] + '_edited'
            metadata.loc[len(metadata)] = list(edited_metadata.values)
            # metadata = metadata.append(edited_metadata, ignore_index=True)
            # We also have to create a file in the dataset folder for the created audio clip, so users can listen to the
            # edited version of the audio clip.
            edited_index_value = int(metadata.index.max())
            edited_file_name = str(edited_index_value) + '_edited.wav'
            metadata.loc[edited_index_value, 'filename'] = edited_file_name
            sf.write(Path(dataset_path, edited_file_name), raw_waveforms, sample_rate)
        else:
            upload_metadata = metadata.loc[0, :].copy()
            filename_noext = filename.split('.wav')[0]
            upload_metadata['label'] = filename_noext + '_uploaded'
            metadata.loc[len(metadata)] = list(upload_metadata.values)
            uploaded_index_value = int(metadata.index.max())
            uploaded_file_name = str(uploaded_index_value) + '_uploaded.wav'
            metadata.loc[uploaded_index_value, 'filename'] = uploaded_file_name
            sf.write(Path(dataset_path, uploaded_file_name), raw_waveforms, sample_rate)


        # We should not be returning anything if we manage to make in place changes. We will see about that.
        return metadata

    def upload_audio(self, audio_content, filename, results, models, metadata, dataset_path, config, dataset):
        """
        Given an audio content upload a file into the local system to be supported by the tool. Also, create
        its transformed representation for all of the models and modify the results and metadata in place.
        :param audio_content: The audio content to be saved locally
        :param filename: The audio filename
        :param results: the results dictionary to be modified
        :param models: The models dictionary to be used for transforming the received audio
        :param metadata: The metadata DataFrame to be modifed with the new sample
        :param dataset_path: The path for storing the new audio sample
        :param config: The Global configuration object.
        :param dataset: Dataset name
        :return: returns the modified metadata
        """
        data = audio_content.encode("utf8").split(b";base64,")[1]
        with open(os.path.join(dataset_path, filename), "wb") as fp:
            fp.write(base64.decodebytes(data))
        raw_waveforms, sample_rate = librosa.load(path=os.path.join(dataset_path, filename),
                                                  sr=config[dataset]['sample_rate'])
        os.remove(os.path.join(dataset_path, filename))
        duration = config[dataset]['duration']
        samples_per_audio = int(duration*sample_rate)
        if len(raw_waveforms) > samples_per_audio:
            diff = len(raw_waveforms) - samples_per_audio
            starting_position = int(diff/2)
            raw_waveforms = raw_waveforms[starting_position:(starting_position + samples_per_audio)]

        elif len(raw_waveforms) < samples_per_audio:
            diff = samples_per_audio - len(raw_waveforms)
            starting_position = int(diff / 2)
            new_waveforms = np.zeros(samples_per_audio)
            new_waveforms[starting_position:(starting_position+len(raw_waveforms))] = raw_waveforms
            raw_waveforms = new_waveforms.copy()
        else:
            pass

        return self.predict_and_add_new_audio(config, raw_waveforms, models, results, metadata, dataset_path, dataset,
                                              selected_index=None, sample_rate=sample_rate, filename=filename)