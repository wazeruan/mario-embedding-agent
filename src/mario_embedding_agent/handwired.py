"""A screenshot-only Mario policy whose every weight is written by hand, not trained.

Pixels -> 1x1 colour-matching ReLU convs -> 4x4 average-pooled cell maps ->
coincidence convs ("Mario here AND obstacle/gap just ahead") -> global max pool ->
a small recurrent ReLU circuit that times the jump button. Buttons: right+B always,
A when the circuit fires. No game RAM, score or position enters the network.
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .environment import make_env, step
from .rewards import StageDetector

CELL = 4                                   # pixels per cell side
SKY = (104, 136, 252)
MARIO = [(248, 56, 0), (172, 124, 0)]      # red cap/shirt, olive hair/shoes
SOLID = [(228, 92, 16), (240, 208, 176),    # ground, bricks, stairs, Goombas
         (136, 20, 0)]                      # block shading
LIGHT_GREEN, DARK_GREEN, WHITE, SKIN = (184, 248, 24), (0, 168, 0), (252, 252, 252), (252, 160, 68)
# A pipe is the only scenery with both greens in one cell: bushes are light green only and
# hills dark green only. A Koopa is the only sprite mixing dark green with white (clouds are
# white without green). Plain clouds, hills, bushes and Q-block orange stay background.
PIPE_REACH = 3                             # cells a mixed-green cell spreads across its pipe
MARIO_REACH = 4                            # cells around a red+olive pair that belong to Mario
KOOPA_REACH = 2                            # cells a green+white cell spreads across its Koopa
TOLERANCE = 12.0                           # L1 colour distance that still counts as a match
GROUND_ROWS = (208 // CELL, 216 // CELL)   # rows of cells holding the ground surface
LOW = (1, 4, 1, 10)                        # rows above feet, cells ahead: 4..16 px up, 4..40 px ahead
TALL = (5, 10, 1, 8)                       # 20..44 px up, 4..32 px ahead; misses blocks floating 48 px up
GAP_AHEAD = (1, 3)                         # pit window: 4..16 px ahead, so take-off is at the edge
HOLD_LOW, HOLD_TALL = 16, 22               # frames A is held: Goomba jump, full jump over pipes/pits
LIFTOFF = 4                                # frames Mario may still touch the ground after pressing A
GATE = 64.0                                # AND-gate weight: exceeds any count of window cells


def _step_unit(x, threshold, width):
    """ReLU pair: 0 below threshold, 1 above threshold+width, linear in between."""
    return F.relu((x-threshold)/width) - F.relu((x-threshold)/width - 1)


class HandWiredPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        colours = [SKY, *MARIO, *SOLID, LIGHT_GREEN, DARK_GREEN, WHITE, SKIN]
        # Layer 1 (1x1 conv, 3 -> 6 per colour): +(x-c) and -(x-c) per channel.
        w1 = torch.zeros(6*len(colours), 3, 1, 1)
        b1 = torch.zeros(6*len(colours))
        for k, colour in enumerate(colours):
            for ch in range(3):
                w1[6*k+2*ch, ch], b1[6*k+2*ch] = 1.0, -colour[ch]
                w1[6*k+2*ch+1, ch], b1[6*k+2*ch+1] = -1.0, colour[ch]
        # Layer 2 (1x1 conv): match_k = ReLU(1 - L1 distance / tolerance).
        w2 = torch.zeros(len(colours), 6*len(colours), 1, 1)
        for k in range(len(colours)):
            w2[k, 6*k:6*k+6] = -1.0/TOLERANCE
        b2 = torch.ones(len(colours))
        # Layer 3 (1x1 conv): [sky, red, olive, solid, light green, dark green, white, skin].
        w3 = torch.zeros(8, len(colours), 1, 1)
        solid = 1+len(MARIO)
        w3[0, 0], w3[1, 1], w3[2, 2], w3[3, solid:solid+len(SOLID)] = 1.0, 1.0, 1.0, 1.0
        for k in range(4):
            w3[4+k, len(colours)-4+k] = 1.0
        b3 = torch.zeros(8)
        for name, value in dict(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3).items():
            self.register_buffer(name, value)

        self.register_buffer('stripe_kernel', torch.ones(1, 1, 3, 1))
        # Feet: a Mario cell with no Mario cell below it (kernel over rows r, r+1).
        self.register_buffer('feet_kernel', torch.tensor([1.0, -1.0]).view(1, 1, 2, 1))
        # Coincidence kernels: Mario's feet at the anchor AND a solid cell in a window ahead of them.
        # Anchor at (row 12, col 10) of a 13 x 21 kernel; channel 0 = solid, 1 = feet.
        for name, (up0, up1, ahead0, ahead1) in dict(low=LOW, tall=TALL).items():
            kernel = torch.zeros(1, 2, 13, 21)
            kernel[0, 1, 12, 10] = GATE
            kernel[0, 0, 12-up1:12-up0+1, 10+ahead0:10+ahead1+1] = 1.0
            self.register_buffer(f'{name}_kernel', kernel)
        gap = torch.zeros(1, 2, 1, 21)
        gap[0, 1, 0, 10] = GATE
        gap[0, 0, 0, 10+GAP_AHEAD[0]:10+GAP_AHEAD[1]+1] = 1.0
        self.register_buffer('gap_kernel', gap)
        # Support: the cell below is solid, unless this cell and the one above are solid too
        # (a wall beside Mario, e.g. the pipe he is jumping along, is not a floor).
        self.register_buffer('wall_kernel', torch.tensor([1.0, 1.0]).view(1, 1, 2, 1))   # rows r-1, r
        self.register_buffer('below_kernel', torch.tensor([0.0, 1.0]).view(1, 1, 2, 1))  # rows r, r+1

    def features(self, frame):
        x = torch.as_tensor(np.ascontiguousarray(frame), dtype=torch.float32).permute(2, 0, 1)[None, :, 32:]  # drop HUD
        h = F.relu(F.conv2d(x, self.w1, self.b1))
        h = F.relu(F.conv2d(h, self.w2, self.b2))
        h = F.relu(F.conv2d(h, self.w3, self.b3))             # [1, 8, 208, 256]
        cells = F.avg_pool2d(h, CELL)                          # [1, 8, 52, 64]
        sky = cells[:, 0:1]
        red, olive = _step_unit(cells[:, 1:2], 0.03, 0.03), _step_unit(cells[:, 2:3], 0.03, 0.03)
        # Mario has red and olive within a few pixels of each other; a mushroom has no olive.
        overlap = F.relu(F.max_pool2d(red, 3, 1, 1) + F.max_pool2d(olive, 3, 1, 1) - 1.0)
        near_mario = F.max_pool2d(overlap, MARIO_REACH*2+1, 1, MARIO_REACH)
        mario = F.relu(near_mario + _step_unit(cells[:, 1:2] + cells[:, 2:3], 0.1, 0.1) - 1.0)
        light, dark = _step_unit(cells[:, 4:5], 0.05, 0.05), _step_unit(cells[:, 5:6], 0.05, 0.05)
        mixed = F.relu(light + dark - 1.0)                     # AND: both greens in one cell
        # Pipe stripes are vertical: AND of three stacked mixed cells rejects bush/hill seams.
        striped = F.relu(F.conv2d(F.pad(mixed, (0, 0, 1, 1)), self.stripe_kernel) - 2.0)
        near_mixed = F.max_pool2d(striped, (3, 2*PIPE_REACH+1), 1, (1, PIPE_REACH))
        pipe = F.relu(near_mixed + torch.clamp(light + dark, max=1.0) - 1.0)
        white, skin = _step_unit(cells[:, 6:7], 0.05, 0.05), _step_unit(cells[:, 7:8], 0.05, 0.05)
        shell = F.relu(dark + white - 1.0)                     # AND: green shell with white
        near_shell = F.max_pool2d(shell, 2*KOOPA_REACH+1, 1, KOOPA_REACH)
        koopa = F.relu(near_shell + torch.clamp(dark + white + skin, max=1.0) - 1.0)
        solid = torch.clamp(_step_unit(cells[:, 3:4], 0.2, 0.1) + pipe + koopa, max=1.0)
        # Closing (dilate, then erode) fills the one-cell seams black block outlines leave.
        solid = -F.max_pool2d(-F.max_pool2d(solid, 3, 1, 1), 3, 1, 1)
        top = 32 // CELL                                       # cell rows are offset by the HUD
        gap_row = _step_unit(sky[:, :, GROUND_ROWS[0]-top:GROUND_ROWS[1]-top].mean(2, keepdim=True), 0.7, 0.2)
        feet = F.relu(F.conv2d(F.pad(mario, (0, 0, 0, 1)), self.feet_kernel))
        feet_column = F.max_pool2d(feet, (feet.shape[2], 1))
        both = torch.cat([solid, feet], 1)
        # Pad so a kernel anchored on any cell sees cells beyond the frame as empty.
        low = F.conv2d(F.pad(both, (10, 10, 12, 0)), self.low_kernel)
        tall = F.conv2d(F.pad(both, (10, 10, 12, 0)), self.tall_kernel)
        gap = F.conv2d(F.pad(torch.cat([gap_row, feet_column], 1), (10, 10, 0, 0)), self.gap_kernel)
        wall = F.relu(F.conv2d(F.pad(solid, (0, 0, 1, 0)), self.wall_kernel) - 1.0)
        support = F.relu(F.conv2d(F.pad(solid, (0, 0, 0, 1)), self.below_kernel) - wall)
        gate = lambda x: _step_unit(F.relu(x - GATE).amax(), 0.5, 0.5)
        return {
            'low': gate(low), 'tall': gate(tall), 'gap': gate(gap),
            'mario': mario.amax(), 'feet_map': feet, 'support_map': support,
        }

    def forward(self, frame, state):
        """state = (a_prev, hold_count, hold_limit, previous feet map); None starts an episode.

        Returns (press_A, new_state, features).
        """
        f = self.features(frame)
        if state is None:
            state = (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.zeros_like(f['feet_map']))
        a_prev, count, limit, feet_prev = state
        # Grounded: feet in the same cell as last frame (no vertical motion) AND support below.
        feet = f.pop('feet_map')
        settled = F.relu(feet + feet_prev + f.pop('support_map') - 2.0)
        f['grounded'] = _step_unit(settled.amax(), 0.0, 1.0)
        high = torch.clamp(f['tall'] + f['gap'], max=1.0)
        trigger = torch.clamp(high + f['low'], max=1.0)
        start = F.relu(trigger + f['grounded'] + (1-a_prev) - 2.0)          # AND of three
        below = F.relu(limit-count) - F.relu(limit-1-count)                  # count < limit
        # Release A on landing (after the take-off frames) so the next jump can start at once.
        early = F.relu(LIFTOFF-count) - F.relu(LIFTOFF-1-count)              # count < LIFTOFF
        airborne_or_early = torch.clamp(1-f['grounded'] + early, max=1.0)
        keep = F.relu(a_prev + below + airborne_or_early - 2.0)              # AND of three
        press = torch.clamp(start + keep, max=1.0)
        big = HOLD_TALL + 2.0                                                # gates multiplications
        target = HOLD_LOW + (HOLD_TALL-HOLD_LOW)*high
        # Hold limit: the target at take-off, raised (never lowered) if a taller obstacle shows up
        # while A is still held: max(limit, target) = limit + ReLU(target - limit).
        limit = F.relu(target - big*(1-start)) + F.relu(limit + F.relu(target-limit) - big*(1-keep))
        count = F.relu(count + 1.0 - big*(1-press))                          # (count+1) if pressed else 0
        return press, (press, count, limit, feet), f


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate', 'play'])
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-start-delay', type=int, default=90,
                        help='idle frames before the first decision, drawn per episode; the emulator '
                             'is deterministic, so this is what varies the timing between episodes')
    parser.add_argument('--max-frames', type=int, default=6000)
    parser.add_argument('--fps', type=float, default=60.0)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    policy = HandWiredPolicy()
    display = None
    if args.command == 'play':
        from .display import Display
        display = Display()
    env = make_env()
    clears = 0
    for episode in range(args.episodes):
        frame, _ = env.reset(seed=args.seed+episode)
        detector = StageDetector()
        detector.observe(frame)
        delay = random.Random(args.seed+episode).randrange(args.max_start_delay+1)
        for _ in range(delay):
            frame, _ = step(env, ())
        state = None
        outcome = 'timeout'
        for t in range(args.max_frames):
            press, state, f = policy(frame, state)
            command = ('right', 'B', 'A') if press > 0.5 else ('right', 'B')
            frame, died = step(env, command)
            if display is not None:
                lines = [f'hand-written network, frame {t}', f'jump: {int(press)}'] + [
                    f'{k}: {float(v):.2f}' for k, v in f.items()]
                if not display.draw(frame, None, lines):
                    return
                time.sleep(1/args.fps)
            if detector.observe(frame):
                outcome = 'clear'
                break
            if died:
                outcome = 'death'
                break
        clears += outcome == 'clear'
        print(f'episode {episode}: start delay {delay}, {outcome} after {t+1} frames', flush=True)
    print(f'cleared {clears}/{args.episodes}')
    if display is not None:
        display.close()


if __name__ == '__main__':
    main()
