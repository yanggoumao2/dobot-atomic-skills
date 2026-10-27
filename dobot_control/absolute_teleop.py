"""Non-actuating absolute joint target planner, with per-arm slow alignment."""
import numpy as np


class AbsoluteTeleopPlanner:
    def __init__(self, align_speed=0.15, follow_speed=4.0, max_step=0.35,
                 tolerance=0.02, settle_samples=3, max_dt=0.05):
        values = (align_speed, follow_speed, max_step, tolerance, max_dt)
        if not all(np.isfinite(v) and v > 0 for v in values) or align_speed > follow_speed:
            raise ValueError("Positive finite planner settings required; align speed <= follow speed")
        if settle_samples < 1:
            raise ValueError("settle_samples must be positive")
        self.align_speed, self.follow_speed = align_speed, follow_speed
        self.max_step, self.tolerance = max_step, tolerance
        self.settle_samples, self.max_dt = settle_samples, max_dt
        self.reset()

    def reset(self):
        self.active = np.zeros(2, dtype=bool)
        self.aligned = np.zeros(2, dtype=bool)
        self.settled = np.zeros(2, dtype=int)

    @staticmethod
    def vector(value):
        result = np.asarray(value, dtype=float)
        if result.shape != (14,) or not np.isfinite(result).all():
            raise ValueError("Expected finite 14-D joint sample")
        return result

    def enable(self, flags, measured, previous):
        flags = np.asarray(flags, dtype=bool)
        if flags.shape != (2,):
            raise ValueError("Expected two Nova flags")
        measured, previous = self.vector(measured), self.vector(previous).copy()
        new = flags & ~self.active
        self.aligned[~flags | new] = False
        self.settled[~flags | new] = 0
        for side in np.flatnonzero(new):
            arm = slice(side * 7, side * 7 + 6)
            previous[arm] = measured[arm]
        self.active = flags.copy()
        return previous

    def step(self, target, measured, previous, elapsed):
        target, measured, previous = map(self.vector, (target, measured, previous))
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Positive finite command interval required")
        result = previous.copy()
        transitioned = []
        for side in np.flatnonzero(self.active):
            arm = slice(side * 7, side * 7 + 6)
            # Use calibrated absolute angles verbatim. FK equivalence modulo
            # 2*pi does not prove a bounded physical joint can take that route.
            side_target = target
            if not self.aligned[side]:
                reached = (np.max(np.abs(side_target[arm] - measured[arm])) <= self.tolerance
                           and np.max(np.abs(side_target[arm] - previous[arm])) <= self.tolerance)
                self.settled[side] = self.settled[side] + 1 if reached else 0
                if self.settled[side] >= self.settle_samples:
                    self.aligned[side] = True
                    transitioned.append(int(side))
            speed = self.follow_speed if self.aligned[side] else self.align_speed
            step_limit = min(self.max_step, speed * min(elapsed, self.max_dt))
            delta = side_target[arm] - previous[arm]
            # Common interpolation factor preserves the straight joint-space
            # path during alignment, without overshoot or instant snapping.
            scale = min(1.0, step_limit / max(float(np.max(np.abs(delta))), 1e-12))
            result[arm] += delta * scale
        return result, transitioned


class CommandInterval:
    """Elapsed time between command proposals that were actually dispatched."""
    def __init__(self):
        self.previous = None

    def reset(self, now):
        self.previous = float(now)

    def elapsed(self, now):
        if self.previous is None:
            return 0.001
        return max(float(now) - self.previous, 0.001)

    def committed(self, now):
        self.previous = float(now)
