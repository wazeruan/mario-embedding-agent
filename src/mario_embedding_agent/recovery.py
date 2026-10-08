"""Conservative pixel-only controller for the underground side-entry pipe."""
import numpy as np
from .rewards import GapDetector


class PipeExitRecovery:
    """Move off the pipe, land, then enter its left-facing opening.

    This is an explicit controller override, not a learned PPO action. Geometry
    is re-detected on every decision; no RAM, coordinates from info, or timer.
    """
    @staticmethod
    def mario(frame):
        # Tolerate screenshot colour-profile rounding, but reject ambiguous sprites.
        pixels = frame.astype(np.int16)
        red = np.max(np.abs(pixels-GapDetector.RED.astype(np.int16)), axis=2) <= 3
        red[:48] = False
        ys, xs = np.where(red)
        if len(xs) < 5 or xs.max()-xs.min() > 24 or ys.max()-ys.min() > 40:
            return None
        x0, x1 = max(0, xs.min()-6), min(256, xs.max()+7)
        y0, y1 = max(48, ys.min()-8), min(216, ys.max()+14)
        crop = pixels[y0:y1, x0:x1]
        masks = [np.max(np.abs(crop-colour.astype(np.int16)), axis=2) <= 3
                 for colour in (GapDetector.RED, GapDetector.SKIN, GapDetector.OLIVE)]
        if masks[1].sum() < 3:
            return None
        sy, _ = np.where(np.logical_or.reduce(masks))
        return float((xs.min()+xs.max())/2), int(y0+sy.max()+1)

    @staticmethod
    def geometry(frame):
        if frame.shape != (240, 256, 3):
            return None
        black = np.all(frame[48:160, 32:192] == 0, axis=2).mean()
        if black < .45:
            return None
        green = (frame[:, :, 1] > 120) & (frame[:, :, 0] < 190) & (frame[:, :, 2] < 80)
        walls = np.flatnonzero(green[48:208, 224:].mean(0) > .85)
        if not len(walls):
            return None
        wall = int(walls[0]+224)
        # The low horizontal pipe sits immediately left of the tall green wall.
        rows = np.flatnonzero(green[160:208, 192:wall].sum(1) >= 12)
        if not len(rows):
            return None
        top = int(rows[0]+160)
        cols = np.flatnonzero(green[top:208, 192:wall].mean(0) > .55)
        if not len(cols):
            return None
        left = int(cols[0]+192)
        mario = PipeExitRecovery.mario(frame)
        if mario is None or mario[0] < left-48:
            return None
        return left, top, 208, mario

    def choose(self, frame):
        geometry = self.geometry(frame)
        if geometry is None:
            return None
        left, top, floor, (x, feet) = geometry
        if feet < floor-3:
            # Retreat far enough that Mario's full body clears the pipe edge.
            if x > left-10:
                return 'left', 16, 'pipe_retreat'
            return 'idle', 8, 'pipe_land'
        return 'right', 24, 'pipe_enter'
