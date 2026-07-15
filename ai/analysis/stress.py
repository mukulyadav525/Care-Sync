class StressAnalyzer:

    def calculate(self,
                  hrv,
                  heart_rate):

        score = 0

        score += max(

            0,

            60 - hrv["rmssd"]

        )

        score += max(

            0,

            heart_rate - 70

        )

        score = min(score,100)

        if score < 30:

            level = "Low"

        elif score < 60:

            level = "Moderate"

        else:

            level = "High"

        return {

            "stress_score":

                round(score),

            "stress_level":

                level

        }