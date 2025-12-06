# data_generator.py
import numpy as np
import pandas as pd
from pathlib import Path
import random
import datetime

OUT = Path("data/dataset.csv")
OUT.parent.mkdir(exist_ok=True)

def generate(n_users=200, days_per_user=180, seed=42):
    random.seed(seed); np.random.seed(seed)
    users = []
    start_date = datetime.date(2024,1,1)
    user_ids = [f"user_{i}" for i in range(1, n_users+1)]
    rows = []
    for u in user_ids:
        # random base profile per user
        age = random.randint(18,65)
        sex = random.choice(["M","F"])
        height_cm = random.randint(150,190) if sex=="M" else random.randint(145,178)
        weight0 = np.round(np.random.normal(75 if sex=="M" else 65, 12),1) # kg
        bmi0 = weight0 / ((height_cm/100)**2)
        # daily noise and habit pattern per user
        baseline_steps = int(np.clip(np.random.normal(6000,2000), 1000, 15000))
        baseline_active = int(np.clip(np.random.normal(30,15), 0, 180)) # active minutes
        baseline_cal_intake = int(np.clip(np.random.normal(2200 if sex=="M" else 1900, 300), 1000, 3500))
        for d in range(days_per_user):
            date = start_date + datetime.timedelta(days=d)
            # simulate weekly pattern + random events
            dow = date.weekday()
            steps = int(np.clip(np.random.normal(baseline_steps + (500 if dow<5 else -800), 2000), 0, 40000))
            active_min = int(np.clip(np.random.normal(baseline_active + (5 if dow<5 else -10), 12), 0, 300))
            sleep_h = round(np.clip(np.random.normal(7 + (-0.3 if dow<5 else 0.5), 0.9), 3, 11),2)
            calories_in = int(np.clip(np.random.normal(baseline_cal_intake + (50 if dow<5 else -100), 250), 1000, 4500))
            # estimate calories burned (simple model: BMR + activity)
            # compute BMR via Mifflin-St Jeor (approx)
            if sex=="M":
                bmr = 10*weight0 + 6.25*height_cm - 5*age + 5
            else:
                bmr = 10*weight0 + 6.25*height_cm - 5*age - 161
            # activity calories roughly: active_min * MET*weight_kgs/60 ; assume MET 4 for average active
            activity_cals = active_min * 4 * weight0 / 60.0
            calories_burned = int(np.clip(bmr*0.55 + activity_cals + np.random.normal(0,100), 1200, 5000))
            # weight evolves slowly toward calorie balance (very simplified physics)
            # delta_weight (kg) = (calories_in - calories_burned) / 7700 per day with some smoothing
            delta_w = (calories_in - calories_burned) / 7700.0
            # make weight time-dependent
            if d==0:
                weight = weight0
            else:
                # small smoothing, some weekly trend
                weight = prev_weight + 0.8*delta_w + np.random.normal(0, 0.05)
            bmi = weight / ((height_cm/100)**2)
            # habit flags
            drank_alcohol = int(np.random.binomial(1, 0.1 if dow<5 else 0.25))
            smoked = int(np.random.binomial(1, 0.05))
            rows.append({
                "user_id": u,
                "date": date.isoformat(),
                "age": age,
                "sex": sex,
                "height_cm": height_cm,
                "weight_kg": round(weight,2),
                "bmi": round(bmi,2),
                "steps": steps,
                "active_min": active_min,
                "sleep_h": sleep_h,
                "calories_in": calories_in,
                "calories_burned": calories_burned,
                "drank_alcohol": drank_alcohol,
                "smoked": smoked
            })
            prev_weight = weight

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("Saved", OUT, "shape=", df.shape)
    return df

if __name__ == "__main__":
    generate(n_users=200, days_per_user=180)
