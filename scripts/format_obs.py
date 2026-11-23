import datetime
import pickle
from pathlib import Path
from typing import Dict

import numpy as np


def save_frame(
    folder: str,
    timestamp: int,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
) -> None:
    payload = dict(obs)
    payload["control"] = np.asarray(action).copy()
    folder_path = Path(folder)
    folder_path.mkdir(exist_ok=True, parents=True)
    recorded_file = folder_path / f"{timestamp}.pkl"

    with open(recorded_file, "wb") as f:
        pickle.dump(payload, f)

def save_action(recorded_file,action: np.ndarray):
    with open(recorded_file, "ab") as f:
        pickle.dump(action, f)


if __name__ == "__main__":
    # test write
    act = [1,2,3,4,5,6]



