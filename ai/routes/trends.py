from fastapi import APIRouter
import pandas as pd
from pathlib import Path

router = APIRouter()


@router.get("/trends")
def get_trends():

    BASE_DIR = Path(__file__).resolve().parents[1]

    csv_path = BASE_DIR / "data" / "sample_ppg.csv"

    df = pd.read_csv(csv_path)

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Create day/hour if missing
    if "day" not in df.columns:
        df["day"] = df["timestamp"].dt.day_name()

    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    heart_rate = []
    stress = []

    for day in day_order:

        day_df = df[df["day"] == day]

        if day_df.empty:
            continue

        hr_points = []
        stress_points = []

        for hour in range(24):

            hour_df = day_df[day_df["hour"] == hour]

            if hour_df.empty:
                continue

            hr_points.append({
                "hour": hour,
                "value": round(hour_df["heart_rate"].mean(), 2)
            })

            stress_points.append({
                "hour": hour,
                "value": round(hour_df["stress_score"].mean(), 2)
            })

        heart_rate.append({
            "day": day,
            "data": hr_points
        })

        stress.append({
            "day": day,
            "data": stress_points
        })

    return {
        "heart_rate": heart_rate,
        "stress": stress
    }