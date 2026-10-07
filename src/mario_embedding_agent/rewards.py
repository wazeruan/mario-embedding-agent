"""Level transition detection from the rendered WORLD label, never game info."""
import numpy as np


class StageDetector:
    def __init__(self):
        self.baseline = None
        self.candidate = None
        self.count = 0

    def observe(self, frame):
        white = np.all(frame >= 235, axis=2)
        label = white[24:32, 152:176]
        # WORLD's numeric label is digit, hyphen, digit in the original RGB HUD.
        if (label.shape != (8, 24) or label[:, :8].sum() < 5 or
                label[:, 16:].sum() < 5 or not label[3:5, 9:15].all()):
            self.candidate, self.count = None, 0
            return False
        key = label.tobytes()
        if self.baseline is None:
            self.baseline = key
        if key == self.baseline:
            self.candidate, self.count = None, 0
            return False
        self.count = self.count + 1 if key == self.candidate else 1
        self.candidate = key
        if self.count >= 3:
            self.baseline = key
            self.candidate, self.count = None, 0
            return True
        return False


def reward(death, cleared):
    return -1000.0 if death else 1.0 if cleared else -1.0
