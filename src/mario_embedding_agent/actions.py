"""Controller actions are button combinations with explicit emulator-frame durations."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Action:
    name: str
    buttons: tuple[str, ...]
    hold_frames: int = 4
    release_frames: int = 0

    @property
    def description(self):
        if 'A' in self.buttons:
            length = 'short low' if self.hold_frames == 4 else 'long high'
            direction = 'right' if 'right' in self.buttons else 'left' if 'left' in self.buttons else 'vertically'
            speed = ' while running' if 'B' in self.buttons else ''
            return (f'Mario jumps {direction}{speed}, using a {length} jump: hold jump for '
                    f'{self.hold_frames} game frames, then release jump for {self.release_frames} frames.')
        return {
            'idle': 'Release all buttons and wait briefly.',
            'right': 'Walk to the right.', 'run_right': 'Run to the right.',
            'left': 'Walk to the left.', 'run_left': 'Run to the left.',
            'down': 'Press down to crouch or enter a pipe.',
            'up': 'Press up to climb a vine.',
            'fire': 'Tap the fire button when Mario has a fire flower.',
        }[self.name]

    def commands(self):
        held = () if self.buttons == ('NOOP',) else self.buttons
        released = tuple(button for button in held if button != ('A' if 'A' in held else 'B'))
        return (held,) * self.hold_frames + (released,) * self.release_frames


ACTIONS = tuple([Action('idle', ('NOOP',)), Action('right', ('right',)),
                 Action('run_right', ('right', 'B')), Action('left', ('left',)),
                 Action('run_left', ('left', 'B')), Action('down', ('down',)),
                 Action('up', ('up',)), Action('fire', ('B',), 4, 2)] + [
    Action(f'{name}_jump_{length}', buttons + ('A',), duration, 2)
    for name, buttons in [('vertical', ()), ('right', ('right',)), ('run_right', ('right', 'B')),
                          ('left', ('left',)), ('run_left', ('left', 'B'))]
    for length, duration in [('short', 4), ('long', 20)]
])


def action_manifest():
    return [{**asdict(action), 'buttons': list(action.buttons), 'description': action.description}
            for action in ACTIONS]
