import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root (finds it automatically)
load_dotenv()

# API key from .env file
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# pathways based on the gnn_vol folder
DATA_DIR = Path("gnn_vol/data")
INTRADAY_DIR = DATA_DIR / "intraday"
RV_DIR = DATA_DIR / "rv"
GRAPH_DIR = DATA_DIR / "graphs"
MODEL_DIR = DATA_DIR / "models"

# Create folders if they don't exist
for folder in [INTRADAY_DIR, RV_DIR, GRAPH_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Realised volatility
INTRADAY_INTERVAL = "5min"          # 5-minute bars from Alpha Vantage
TRADING_MINUTES_PER_DAY = 390
BARS_PER_DAY = 78                   # 390 / 5


# HAR features ---> Needed for calculations of RV
HAR_DAILY_LAG = 1                   # yesterday
HAR_WEEKLY_LAG = 5                  # 5-day average
HAR_MONTHLY_LAG = 22                # 22-day average


# Graph (GLASSO)
GLASSO_ALPHA = None                 # None = use cross-validation to pick best penalty
# max iterations for GLASSO convergence (could be more, 200 is starting point)
GLASSO_MAX_ITER = 200


# GNN model
GNN_HIDDEN_DIM = 9                  # neurons in the GNN layer (based on paper)
GNN_N_LAYERS = 1                    # 1 layer = 1-hop neighbours only
# train 5 models with different seeds --> creates a reliable average
GNN_N_ENSEMBLE = 5
GNN_LEARNING_RATE = 1e-3            # Adam optimiser learning rate
GNN_BATCH_SIZE = 32                 # mini-batch size for training
# max loops through data set for training. Should stop sooner in all cases
GNN_MAX_EPOCHS = 200
# stop if validation loss doesn't improve for this many loops
GNN_EARLY_STOP_PATIENCE = 15
GNN_LOSS = "qlike"
# clamp predictions above this to avoid log(0) in QLIKE
GNN_PRED_FLOOR = 1e-6


# Rolling refit
TRAIN_WINDOW = 1000                 # trading days of history to use (~4 years)
# last 25% of train window used for validation / early stopping
VALIDATION_SPLIT = 0.25
REFIT_FREQUENCY = 21                # retrain everything every 21 trading days


# Forecast
FORECAST_HORIZONS = [1, 5]          # 1-day and 5-day ahead
# trading days per year (used to convert daily RV to annual vol)
ANNUALISATION_FACTOR = 252


# Alpha Vantage rate limits (just to allow rv_compute.py to call data in a way that it is not rejected by the API)
# Kept here so it can easily be removed if we get a better API
API_CALLS_PER_MINUTE = 5
API_DELAY_SECONDS = 13
