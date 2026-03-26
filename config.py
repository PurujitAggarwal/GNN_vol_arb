import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://data.alpaca.markets/v2"

DATA_DIR = Path("gnn_vol/data")
INTRADAY_DIR = DATA_DIR / "intraday"
RV_DIR = DATA_DIR / "rv"
GRAPH_DIR = DATA_DIR / "graphs"
MODEL_DIR = DATA_DIR / "models"

for folder in [INTRADAY_DIR, RV_DIR, GRAPH_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


INTRADAY_INTERVAL = "5Min"
BARS_PER_DAY = 78                   # 390 trading minutes / 5

HAR_DAILY_LAG = 1
HAR_WEEKLY_LAG = 5
HAR_MONTHLY_LAG = 22

GLASSO_ALPHA = 0.0002               # Fixed alpha — ~50x faster than CV
GLASSO_MAX_ITER = 200

GNN_HIDDEN_DIM = 9
GNN_N_LAYERS = 1
GNN_N_ENSEMBLE = 5
GNN_LEARNING_RATE = 1e-3
GNN_BATCH_SIZE = 64
GNN_MAX_EPOCHS = 50
GNN_EARLY_STOP_PATIENCE = 10
GNN_LOSS = "qlike"
GNN_PRED_FLOOR = 1e-6

TRAIN_WINDOW = 504
VALIDATION_SPLIT = 0.25
REFIT_FREQUENCY = 21                # Match EGARCH refit schedule

FORECAST_HORIZONS = [1, 5]
ANNUALISATION_FACTOR = 252
