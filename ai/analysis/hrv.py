import numpy as np


class HRVAnalyzer:

    def rmssd(self, rr):

        diff = np.diff(rr)

        return np.sqrt(

            np.mean(diff ** 2)

        )

    def sdnn(self, rr):

        return np.std(rr)

    def mean_rr(self, rr):

        return np.mean(rr)

    def mean_hr(self, rr):

        return 60000 / np.mean(rr)

    def pnn50(self, rr):

        diff = np.abs(np.diff(rr))

        return np.sum(diff > 50) / len(diff) * 100

    def calculate(self, rr):

        return {

            "rmssd":

                round(self.rmssd(rr),2),

            "sdnn":

                round(self.sdnn(rr),2),

            "mean_rr":

                round(self.mean_rr(rr),2),

            "mean_hr":

                round(self.mean_hr(rr),2),

            "pnn50":

                round(self.pnn50(rr),2)

        }