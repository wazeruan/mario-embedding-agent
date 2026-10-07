"""Expose screenshots and a death-only termination flag; discard game metadata."""
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
from .actions import ACTIONS


class PixelMario(SuperMarioBrosEnv):
    def _get_reward(self):
        return 0.0

    def _get_info(self):
        return {}

    def _get_truncated(self):
        return False

    def _get_terminated(self):
        # This is the sole permitted state signal, confined to the adapter.
        return bool(self._is_dying or self._is_dead or self._is_game_over)

    def _did_step(self, done):
        pass  # Preserve the actual animation instead of RAM-based frame skipping.

    def _did_reset(self):
        pass


COMMANDS = tuple(dict.fromkeys(command for action in ACTIONS for command in action.commands()))


def make_env():
    return JoypadSpace(PixelMario(), [list(command) or ['NOOP'] for command in COMMANDS])


def step(env, command):
    frame, _, terminated, _, _ = env.step(COMMANDS.index(command))
    return frame.copy(), bool(terminated)
