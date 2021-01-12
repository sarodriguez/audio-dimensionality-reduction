from pathlib import Path
import pandas as pd
import numpy as np
import os
import librosa
# from torch.utils.data import Dataset
#
# class FaceLandmarksDataset(Dataset):
#     """Face Landmarks dataset."""
#
#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.landmarks_frame)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample

class AudioWaveDataset:
    """
    Class to handle a wave dataset (.wav files) as a numpy matrix, a prerequisite is to have same length clips.
    """
    def __init__(self, path, name_as_metadata=False, name_separator='_'):
        self.path = Path(path)
        filenames = os.listdir(self.path)
        self.audio = []
        self.filename_data = []
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                audio_path = Path(path, filename)
                y, sr = librosa.load(audio_path, sr=None)
                self.audio.append(y)
                if name_as_metadata:
                    name_split = filename.lower().replace('.wav', '').split(name_separator)
                    name_split.append(filename)
                    self.filename_data.append(name_split)

    def get_data(self):
        """
        Return a numpy array containing all the audio clips in the Dataset
        :return: np array with audio clips
        """
        return np.array(self.audio)

    def get_metadata(self):
        """
        Return a DataFrame with the metadata for the dataset, extracted from the filenames.
        :return:
        """
        df = pd.DataFrame(self.filename_data)
        columns = list(df.columns)
        columns[-1] = 'filename'
        df.columns = columns

        return df
