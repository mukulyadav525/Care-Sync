import numpy as np

from scipy.signal import find_peaks


class PeakDetector:

    def __init__(self,
                 sampling_rate=64):

        self.fs = sampling_rate

    def detect_peaks(self, signal):

        peaks, _ = find_peaks(

            signal,

            distance=self.fs * 0.45,

            prominence=0.25

        )

        return peaks

    def rr_intervals(self, peaks):

        rr = np.diff(peaks)

        rr = rr / self.fs

        rr = rr * 1000

        return rr

    def heart_rate(self, rr):

        if len(rr) == 0:

            return 0

        return float(60000 / np.mean(rr))