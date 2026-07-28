import pathlib
from pathlib import Path
from dataclasses import dataclass

# Creating class configuration for data paths
@dataclass
class Config:
    # Data paths
    DATA_DIR: Path = Path(__file__).parent.parent / "data"
    OUT_DIR: Path = Path(__file__).parent.parent / "Output"
    # Data split points
    RECENT_WEEKS: int = 8
    VALIDATION_WEEK: str = "2020-36"
    TEST_WEEK: str = "2020-37"
    TRAIN_END_WEEK: str = "2020-35"

    # Candidate generation Parameters
    TOPK_RECENT: int = 30 
    TOPK_SEASONAL: int = 30
    REPURCHASE_N: int = 30
    I2I_TOPK: int = 20
    I2I_WEEKS: int = 8
    I2I_MIN_SALES: int = 5
    I2I_SEEDS: int = 5
    IDX_TOP_N: int = 30

    # Collaborative matrix factorization params
    CF_FACTORS:int = 64
    CF_ITERS: int = 15
    CF_REG:float = 0.01
    CF_WINDOWS: tuple = (8,52)

    # RANKER
    CANDIDATE_CAP: int = 200
    FINAL_K: int = 12

    RANDOM_STATE:int = 42
    N_JOBS:int = -1

cfg = Config()

if __name__ == "__main__":
    print("TESTING CONFIG")
    print("TEST_WEEK:",cfg.TEST_WEEK)
    print("DATA_DIR exists:",cfg.DATA_DIR.exists())
    print("OUT_DIR exists:",cfg.OUT_DIR.exists())

