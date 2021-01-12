import librosa
import soundfile as sf
import yaml
import numpy as np
from scipy import signal
import math
from pathlib import Path
import argparse
from utilities import dir_path

config = yaml.safe_load(open("config.yaml"))


def array_linear_variation(sound_waves: np.array, variation_type: str, lowest_factor=0.0, highest_factor=1.0):
    """
    Apply a linear modification over a received numpy array. This will modify the sound waves gradually following
    the factors specified
    :param sound_waves: Numpy Array with sound waves to modify.
    :param variation_type: categories that specify the type of variation to be used
    :param lowest_factor: Lowest factor, i.e. the lowest modification would be multiplying the sound_waves by this
    factor. If 0 then the sound_wave will be 0 at the lowest point.
    :param highest_factor: highest factor, i.e. the highest modification would be multiplying the sound_waves by this
    factor.
    :return: np.array with the modiifed waves.
    """
    variation_types = ('increase', 'decrease', 'increasedecrease', 'decreaseincrease', None, 'none')
    if variation_type is None or variation_type == 'none':
        return sound_waves
    if variation_type not in variation_types:
        raise ValueError("Not supported variation type")
    if len(sound_waves.shape) > 2:
        raise ValueError("Only support 1D or 2D arrays, where, in 2D, rows represent samples")
    if len(sound_waves.shape) == 1:
        sound_waves = sound_waves.reshape(1, -1)
    length_sound_waves = sound_waves.shape[1]
    if variation_type == 'increase':
        position = (np.arange(length_sound_waves) + 1)
        variation = position / length_sound_waves
    elif variation_type == 'decrease':
        position = (np.arange(length_sound_waves) + 1)
        variation = position[::-1] / length_sound_waves
    elif variation_type == 'increasedecrease':
        variation = np.concatenate((np.arange(math.ceil(length_sound_waves / 2))+1,
                                    np.arange(int(length_sound_waves / 2), 0, -1))) / math.ceil(length_sound_waves / 2)
    else:  # variation_type == 'decreaseincrease':
        variation = np.concatenate((np.arange(int(length_sound_waves / 2), 0, -1),
                                    np.arange(math.ceil(length_sound_waves / 2))+1)) / math.ceil(length_sound_waves / 2)
    # shift and resize variation based on highest and lowest factors
    variation = (variation * (highest_factor - lowest_factor)) + lowest_factor
    mod_sound_waves = sound_waves * variation
    if mod_sound_waves.shape[0] == 1:
        return mod_sound_waves.reshape(-1,)
    else:
        return mod_sound_waves


def create_synthetic_audio_dataset(output_path):
    """
    Follow the specified configurations in the global config file to create an audio dataset with different types of
    waves
    :param output_path: the path to store the resulting audio files.
    :return:
    """
    sample_rate = config['synthetic_dataset']['sample_rate']
    duration = config['synthetic_dataset']['duration']
    frequency_levels = config['synthetic_dataset']['frequency_std_levels']
    wave_types = config['synthetic_dataset']['wave_types']
    samples_per_level = config['synthetic_dataset']['samples_per_level']
    frequency_variations = config['synthetic_dataset']['frequency_variations']
    amplitude_variations = config['synthetic_dataset']['amplitude_variations']
    combination_probability = config['synthetic_dataset']['combination_probability']

    wav_filenames = np.char.array([])
    wav_files = np.array([]).reshape(-1, duration * sample_rate)
    t = np.linspace(0, duration, duration * sample_rate, endpoint=False)
    dt = 1 / sample_rate
    # Create a loop for each of the frequency levels
    for frequency, std in frequency_levels.items():
        sample_frequencies = np.random.normal(frequency, std, samples_per_level)
        f_array = sample_frequencies.reshape(-1, 1).repeat(sample_rate * duration, axis=1)

        # Add frequency variation
        level_frequency_variations = np.random.randint(len(frequency_variations), size=samples_per_level)
        frequency_lowest_factor = np.random.uniform(low=0.7, high=1, size=(samples_per_level, 1))
        for i in range(len(frequency_variations)):
            freq_variation_mask = level_frequency_variations == i
            f_array[freq_variation_mask] = array_linear_variation(f_array[freq_variation_mask],
                                                                  variation_type=frequency_variations[i],
                                                                  lowest_factor=frequency_lowest_factor[freq_variation_mask])

        # Create waves with randomly selected types
        level_wave_types = np.random.randint(len(wave_types), size=samples_per_level)
        wave_input = 2 * np.pi * np.cumsum(f_array, axis=1) * dt
        for i in range(len(wave_types)):
            level_wave_type_mask = level_wave_types == i
            if wave_types[i] == 'sin':
                # f_array[level_wave_type_mask] = np.sin(2 * np.pi * f_array[level_wave_type_mask] * t)
                f_array[level_wave_type_mask] = np.sin(wave_input[level_wave_type_mask])
            elif wave_types[i] == 'square':
                # f_array[level_wave_type_mask] = signal.square(2 * np.pi * f_array[level_wave_type_mask] * t)
                f_array[level_wave_type_mask] = signal.square(wave_input[level_wave_type_mask])
            elif wave_types[i] == 'triangle':
                # f_array[level_wave_type_mask] = signal.sawtooth(2 * np.pi * f_array[level_wave_type_mask] * t,
                #                                                 width=0.5)
                f_array[level_wave_type_mask] = signal.sawtooth(wave_input[level_wave_type_mask], width=0.5)
            else:  # if wave_types[i] == 'sawtooth':
                # f_array[level_wave_type_mask] = signal.sawtooth(2 * np.pi * f_array[level_wave_type_mask] * t)
                f_array[level_wave_type_mask] = signal.sawtooth(wave_input[level_wave_type_mask])

        # Add amplitude variation
        level_amplitude_variations = np.random.randint(len(amplitude_variations), size=samples_per_level)
        amplitude_lowest_factor = np.random.uniform(low=0.0, high=1, size=(samples_per_level, 1))
        for j in range(len(amplitude_variations)):
            ampl_variation_mask = level_amplitude_variations == j
            f_array[ampl_variation_mask] = array_linear_variation(f_array[ampl_variation_mask],
                                                                  variation_type=amplitude_variations[j],
                                                                  lowest_factor=amplitude_lowest_factor[ampl_variation_mask])
        # Combine waves with the specified combination_probability probability
        combination_mask = np.random.rand(samples_per_level) > (1 - combination_probability)

        combine_indexes = np.random.randint(samples_per_level, size=combination_mask.sum())
        f_array[combination_mask] = f_array[combination_mask] + f_array[combine_indexes, :]
        # Create filenames
        name_frequency = np.char.array(frequency_variations)[level_frequency_variations]
        name_amplitude = np.char.array(amplitude_variations)[level_amplitude_variations]
        name_wave_type = np.char.array(wave_types)[level_wave_types]
        name_combination = np.char.array(['single', 'combination'])[combination_mask.astype(int)]
        filenames = '{}Hz'.format(frequency) + \
                    '_' + np.char.array(sample_frequencies.astype(int).astype(str)) + "Hz" + \
                    '-' + np.char.array(np.around(frequency_lowest_factor, decimals=2).reshape(-1,).astype(str)) +\
                    '-' + np.char.array(np.around(amplitude_lowest_factor, decimals=2).reshape(-1,).astype(str)) +\
                    '_' + name_frequency + \
                    '_' + name_amplitude + \
                    '_' + name_wave_type + \
                    '_' + name_combination + '.wav'

        wav_files = np.concatenate([wav_files, f_array], axis=0)
        wav_filenames = np.concatenate([wav_filenames, filenames])

    # Save created files
    for i in range(len(wav_files)):
        sf.write(Path(output_path, "{}_".format(i)+wav_filenames[i]), wav_files[i], sample_rate)
        # import matplotlib.pyplot as plt
        # plt.plot(wav_files)
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a synthetic dataset of wav files. Make sure to check'
                                                 'config.yaml to define the dataset configuration')
    parser.add_argument('--output_path', type=dir_path, required=True,
                        help='Specify the absolute path to store the created dataset')
    args = parser.parse_args()
    create_synthetic_audio_dataset(args.output_path)
    print(args)
