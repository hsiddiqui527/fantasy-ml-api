import joblib
import pandas as pd


model = joblib.load("models/draft_model.pkl")

def predict_adp(position: str, team: str, year: int) -> float:
    # Prepare input as a list of dicts (what the model expects)
    input_df = pd.DataFrame([{
        "position": position,
        "team": team,
        "Year": year
    }])
    
    prediction = model.predict(input_df)
    return round(float(prediction[0]), 2)
