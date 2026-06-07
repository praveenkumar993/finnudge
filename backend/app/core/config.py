import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME        : str  = "FinNudge"
    VERSION         : str  = "1.0.0"
    ENVIRONMENT     : str  = os.getenv("ENVIRONMENT", "development")

    # paths
    MODEL_PATH      : str  = "data/processed/two_tower.pt"
    FAISS_INDEX_PATH: str  = "data/processed/nudge.index"
    NUDGE_META_PATH : str  = "data/processed/nudge_metadata.json"
    NUDGE_FEAT_PATH : str  = "data/processed/nudge_features.json"
    USER_FEAT_PATH  : str  = "data/processed/features.json"
    DB_PATH         : str  = "data/processed/finnudge.db"

    # recommendation
    TOP_K           : int  = 3
    EMBEDDING_DIM   : int  = 64

    # online learning
    LEARNING_RATE   : float = 0.01
    NOISE_DECAY     : float = 0.995

settings = Settings()