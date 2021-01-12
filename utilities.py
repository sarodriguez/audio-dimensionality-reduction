import os
import numpy as np
import librosa

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        '''Log mel feature extractor.

        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        '''

        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)
        self.sample_rate = sample_rate
        self.mel_bins = mel_bins
        self.fmin = fmin

        self.melW = librosa.filters.mel(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax).T
        '''(n_fft // 2 + 1, mel_bins)'''

    def transform(self, audio):
        '''Extract feature of a singlechannel audio file.

        Args:
          audio: (samples,)

        Returns:
          feature: (frames_num, freq_bins)
        '''

        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func

        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio,
            n_fft=window_size,
            hop_length=hop_size,
            window=window_func,
            center=True,
            dtype=np.complex64,
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''

        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)

        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10,
            top_db=None)

        logmel_spectrogram = logmel_spectrogram.astype(np.float32)

        return logmel_spectrogram


    def inverse_transform(self, spectrogram):
        return librosa.feature.inverse.mel_to_audio(spectrogram,
                                                    sr=self.sample_rate, n_fft=self.window_size, #n_mels=self.mel_bins,
                                                    hop_length=self.hop_size,
                                                    center=True,  # fmin=fmin, # fmax=fmax
                                                    )
        # y = audio,
        # n_fft = window_size,
        # hop_length = hop_size,
        # window = window_func,
        # center = True,
        # dtype = np.complex64,
        #
