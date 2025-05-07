from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import HTTPException
from datetime import datetime
from model_utils import predict_adp 
from fastapi import HTTPException
import logging
import pandas as pd
import time

df = pd.read_csv("clean_data/cleaned_players.csv")
valid_teams = set(df['team'].dropna().unique())
valid_years = set(df['Year'].dropna().unique())



# Set up basic logging format
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Define what input we expect from the frontend
class DraftRequest(BaseModel):
    position: str
    team: str
    year: int
    
class TopPicksRequest(BaseModel):
    position: str
    year: int
    team: str = None  # Optional


@app.post("/recommend")
def recommend_player(request: DraftRequest):
    logging.info(f"Received request: {request.dict()}")
    
    alerts = []

    # Year check
    if request.year not in valid_years:
        alert = f"⚠️ Suspicious year input: {request.year}"
        logging.warning(alert)
        alerts.append(alert)

    # Team check
    if request.team.upper() not in valid_teams:
        alert = f"⚠️ Unknown team: {request.team}"
        logging.warning(alert)
        alerts.append(alert)

    # Save alerts to a file (simulate sending to Slack or Webex)
    if alerts:
        with open("anomaly_alerts.log", "a") as f:
            for msg in alerts:
                f.write(f"{msg}\n")
        print(f"ALERT TRIGGERED:\n" + "\n".join(alerts))


    # Validate position
    valid_positions = {"RB", "WR", "QB", "TE"}
    if request.position.upper() not in valid_positions:
        logging.warning(f"Invalid position value: {request.position}")
        raise HTTPException(
            status_code=400,
            detail=f"Position '{request.position}' is not supported. Use one of: {valid_positions}"
        )
        
    try:
        start_time = time.time()
        predicted_adp = predict_adp(request.position, request.team, request.year)
        # time.delay(0.6)  # Simulate some processing time
        elapsed_time = time.time() - start_time
        logging.info(f"Inference time: {elapsed_time:.4f} seconds")

        if elapsed_time > 0.5:
            logging.warning("⚠️ Inference exceeded 500ms — potential latency issue")
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    logging.info(f"Predicted ADP: {predicted_adp}")

    return {
        "position": request.position,
        "team": request.team,
        "year": request.year,
        "predicted_adp": predicted_adp
    }

@app.post("/top-picks")
def get_top_picks(request: TopPicksRequest):
    logging.info(f"Top picks request: {request.dict()}")

    filtered = df[
        (df['position'].str.upper() == request.position.upper()) &
        (df['Year'] == request.year)
    ]

    if request.team:
        filtered = filtered[filtered['team'].str.upper() == request.team.upper()]

    top_players = filtered.sort_values(by='adp').head(3)

    if top_players.empty:
        raise HTTPException(status_code=404, detail="No players found with that criteria.")

    results = top_players[['name', 'position', 'team', 'adp']].to_dict(orient='records')
    logging.info(f"Top Results Response: {results}")
    return {"top_players": results}
