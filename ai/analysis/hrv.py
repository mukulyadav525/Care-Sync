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

        # float(...) before round(...): round() on a bare numpy scalar
        # returns another numpy scalar (repr "np.float64(15.62)"), which
        # leaked into the AI report and chat responses verbatim. Casting to
        # a native Python float first gives a plain "15.62" everywhere
        # downstream (JSON, f-strings, the report template).
        return {

            "rmssd":

                round(float(self.rmssd(rr)), 2),

            "sdnn":

                round(float(self.sdnn(rr)), 2),

            "mean_rr":

                round(float(self.mean_rr(rr)), 2),

            "mean_hr":

                round(float(self.mean_hr(rr)), 2),

            "pnn50":

                round(float(self.pnn50(rr)), 2)

        }