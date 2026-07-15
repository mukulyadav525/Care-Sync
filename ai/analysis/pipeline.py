import numpy as np
import pandas as pd

from ai.analysis.preprocessing import SignalPreprocessor
from ai.analysis.peak_detection import PeakDetector
from ai.analysis.hrv import HRVAnalyzer
from ai.analysis.stress import StressAnalyzer


class HealthPipeline:

    def __init__(self):

        self.preprocessor = SignalPreprocessor()
        self.detector = PeakDetector()
        self.hrv = HRVAnalyzer()
        self.stress = StressAnalyzer()

    def analyze_ppg(self, ppg_signal):

        processed = self.preprocessor.preprocess(ppg_signal)

        peaks = self.detector.detect_peaks(processed)

        rr = self.detector.rr_intervals(peaks)

        heart_rate = self.detector.heart_rate(rr)

        hrv = self.hrv.calculate(rr)

        stress = self.stress.calculate(hrv, heart_rate)

        return {
            "heart_rate": heart_rate,
            "hrv": hrv,
            "stress": stress
        }

    def analyze_csv(self, csv_path):

        df = pd.read_csv(csv_path)

        # Try to automatically locate the PPG column
        possible_columns = [
            "ppg",
            "PPG",
            "bvp",
            "BVP",
            "signal",
            "Signal",
            "value",
            "Value"
        ]

        ppg = None

        for col in possible_columns:
            if col in df.columns:
                ppg = df[col].to_numpy(dtype=float)
                break

        if ppg is None:
            # fallback to first numeric column
            numeric = df.select_dtypes(include="number")

            if numeric.shape[1] == 0:
                raise ValueError("No numeric PPG column found.")

            ppg = numeric.iloc[:, 0].to_numpy(dtype=float)

        return self.analyze_ppg(ppg)