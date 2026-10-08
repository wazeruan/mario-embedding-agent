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


def reward(death, cleared, moved_right=False, gap_bonus=0.0):
    """Raw reward: terminal outcomes override movement and shaping rewards."""
    return -1000.0 if death else 1.0 if cleared else float(bool(moved_right)) + float(gap_bonus)


class GapDetector:
    """Conservative RGB detector for original small-Mario blue-sky overworld.

    Require grounded approach, airborne passage over a visible ground hole,
    and three grounded frames beyond its far edge. Unsupported palettes,
    missing sprites and ambiguous gaps fail closed rather than guessing.
    """
    SKY = np.array((104, 136, 252), dtype=np.uint8)
    RED = np.array((248, 56, 0), dtype=np.uint8)
    SKIN = np.array((252, 160, 68), dtype=np.uint8)
    OLIVE = np.array((172, 124, 0), dtype=np.uint8)

    def __init__(self):
        self.gap = None
        self.airborne = False
        self.landed = 0
        self.completed = False
        self.grounded = 0

    @classmethod
    def mario(cls, frame):
        red = np.all(frame == cls.RED, axis=2)
        red[:48] = False
        ys, xs = np.where(red)
        if len(xs) < 5 or xs.max()-xs.min() > 24 or ys.max()-ys.min() > 30:
            return None
        x0, x1 = max(0, xs.min()-6), min(256, xs.max()+7)
        y0, y1 = max(48, ys.min()-8), min(216, ys.max()+9)
        crop = frame[y0:y1, x0:x1]
        skin = np.all(crop == cls.SKIN, axis=2)
        if skin.sum() < 3:
            return None
        sprite = skin | np.all(crop == cls.RED, axis=2) | np.all(crop == cls.OLIVE, axis=2)
        sy, sx = np.where(sprite)
        return float((xs.min()+xs.max())/2), int(y0+sy.max()+1)

    def observe(self, frame):
        if frame.shape != (240, 256, 3):
            return False
        mario = self.mario(frame)
        # A blue void at the ground row, bounded by visible brick ground.
        holes = np.all(frame[212:224] == self.SKY, axis=2).mean(axis=0) > .9
        ground = np.all(frame[212:224] == (228, 92, 16), axis=2).sum()
        if mario is None or ground < 150:
            self.landed = 0
            self.grounded = 0
            return False
        x, feet = mario
        self.grounded = self.grounded+1 if 205 <= feet <= 210 else 0
        edges = np.diff(np.r_[False, holes, False].astype(int))
        gaps = [(int(a), int(b)) for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])
                if 8 <= b-a <= 96 and a > 0 and b < 256]
        if self.gap is not None:
            matches = [g for g in gaps if abs(g[0]-self.gap[0]) <= 12 and abs(g[1]-self.gap[1]) <= 12]
            if not matches:
                self.gap = None
                self.airborne = self.completed = False
                self.landed = 0
                return False
            self.gap = min(matches, key=lambda g: abs(g[0]-self.gap[0]))
        elif self.grounded >= 3:
            ahead = [g for g in gaps if 4 <= g[0]-x <= 96]
            if ahead:
                self.gap = min(ahead, key=lambda g: g[0])
                self.airborne = self.completed = False
        if self.gap is None or self.completed:
            return False
        left, right = self.gap
        if feet < 201 and left-8 <= x <= right+8:
            self.airborne = True
        self.landed = self.landed+1 if self.airborne and x > right+3 and 205 <= feet <= 210 else 0
        if self.landed >= 3:
            self.completed = True
            return True
        return False


class RightwardProgress:
    """Estimate Mario world displacement from RGB sprite and scrolling scenery.

    Mario screen motion plus leftward scenery scroll approximates world motion.
    Ambiguous sprite/scene matches contribute no estimate.
    """
    def __init__(self):
        self.previous = None
        self.last_x = None
        self.unobserved_scroll = 0.0

    def reset(self, frame):
        self.previous = np.array(frame, copy=True)
        mario = GapDetector.mario(frame)
        self.last_x = None if mario is None else mario[0]
        self.unobserved_scroll = 0.0

    @staticmethod
    def _scene_scroll(previous, current):
        # Compare the textured ground strip; positive d means ground moved left.
        a, b = previous[208:240], current[208:240]
        scores = []
        for d in range(-8, 9):
            if d >= 0:
                old, new = a[:, d:], b[:, :256-d]
            else:
                old, new = a[:, :256+d], b[:, -d:]
            error = float(np.abs(old.astype(np.int16)-new.astype(np.int16)).mean())
            scores.append((error, d))
        scores.sort()
        zero_error = next(error for error, shift in scores if shift == 0)
        best_error, shift = scores[0]
        if shift == 0 or best_error + 1.0 >= zero_error or scores[1][0] - best_error < 0.1:
            return 0.0
        return float(shift)

    def observe(self, frame):
        if frame.shape != (240, 256, 3):
            return None
        if self.previous is None:
            self.reset(frame)
            return None
        scroll = self._scene_scroll(self.previous, frame)
        self.unobserved_scroll += scroll
        self.previous = np.array(frame, copy=True)
        mario = GapDetector.mario(frame)
        if mario is None:
            # Drop an incomplete interval to avoid assigning stale motion later.
            self.last_x = None
            self.unobserved_scroll = 0.0
            return None
        x = mario[0]
        if self.last_x is None:
            self.last_x = x
            self.unobserved_scroll = 0.0
            return None
        progress = x - self.last_x + self.unobserved_scroll
        self.last_x = x
        self.unobserved_scroll = 0.0
        return float(progress)
