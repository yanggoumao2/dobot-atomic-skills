import time
import threading
from typing import Any, Dict, Protocol

import numpy as np


class Agent(Protocol):
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Returns an action given an observation.

        Args:
            obs: observation from the environment.

        Returns:
            action: action to take on the environment.
        """
        raise NotImplementedError

    def set_torque(self, _flag=False):

        raise NotImplementedError

    def get_keys(self):

        raise NotImplementedError


class DummyAgent(Agent):
    def __init__(self, num_dofs: int):
        self.num_dofs = num_dofs

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.zeros(self.num_dofs)


class BimanualAgent(Agent):
    def __init__(self, agent_left: Agent, agent_right: Agent):
        self.agent_left = agent_left
        self.agent_right = agent_right

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        left_obs = {}
        right_obs = {}
        for key, val in obs.items():
            L = val.shape[0]
            half_dim = L // 2
            assert L == half_dim * 2, f"{key} must be even, something is wrong"
            left_obs[key] = val[:half_dim]
            right_obs[key] = val[half_dim:]
        actions = [None, None]

        def _read(index, agent, agent_obs):
            actions[index] = agent.act(agent_obs)

        threads = [
            threading.Thread(
                target=_read,
                args=(0, self.agent_left, left_obs),
                daemon=True,
            ),
            threading.Thread(
                target=_read,
                args=(1, self.agent_right, right_obs),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        aaa = np.concatenate(actions)
        return aaa

    def set_torque(self, which_hand=2,  _flag=False):
        if which_hand == 0:
            self.agent_left.set_torque(_flag)
        elif which_hand == 1:
            self.agent_right.set_torque(_flag)
        else:
            self.agent_left.set_torque(_flag)
            self.agent_right.set_torque(_flag)

    def get_keys(self):
        keys = [None, None]
        errors = [None, None]

        def _read(index, agent):
            try:
                value = np.asarray(agent.get_keys(), dtype=float)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(f"expected three finite button values, got shape={value.shape}")
                keys[index] = value
            except Exception as exc:
                errors[index] = exc

        threads = [
            threading.Thread(target=_read, args=(0, self.agent_left), daemon=True),
            threading.Thread(target=_read, args=(1, self.agent_right), daemon=True),
        ]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 1.5
        for index, thread in enumerate(threads):
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            side = "LEFT" if index == 0 else "RIGHT"
            if thread.is_alive():
                raise RuntimeError(f"{side} master button read timed out")
            if errors[index] is not None:
                raise RuntimeError(f"{side} master button read failed: {errors[index]}") from errors[index]
        return np.stack(keys)
