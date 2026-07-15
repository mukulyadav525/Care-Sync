import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


class SignalPreprocessor:

    def __init__(self, sampling_rate=64):
        self.fs = sampling_rate

    def bandpass_filter(self,
                        signal,
                        lowcut=0.5,
                        highcut=8,
                        order=4):

        nyquist = 0.5 * self.fs

        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = butter(
            order,
            [low, high],
            btype="band"
        )

        return filtfilt(b, a, signal)

    def normalize(self, signal):

        signal = np.asarray(signal)

        return (signal - np.mean(signal)) / np.std(signal)

    def smooth(self, signal):

        window = min(21, len(signal))

        if window % 2 == 0:
            window -= 1

        if window < 5:
            return signal

        return savgol_filter(
            signal,
            window_length=window,
            polyorder=3
        )

    def preprocess(self, signal):

        filtered = self.bandpass_filter(signal)

        normalized = self.normalize(filtered)

        smoothed = self.smooth(normalized)

        return smoothed