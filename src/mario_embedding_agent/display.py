"""Visible game and the exact screenshot used by the policy."""
import pygame
import numpy as np


class Display:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((1100, 720))
        pygame.display.set_caption('Mario — visual encoder + PPO')
        self.font = pygame.font.SysFont('Menlo', 16)

    def draw(self, frame, policy_frame, lines):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
        self.window.fill((18, 22, 30))
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self.window.blit(pygame.transform.scale(surface, (768, 720)), (0, 0))
        for i, line in enumerate(lines[:23]):
            self.window.blit(self.font.render(str(line), True, (225, 232, 242)), (782, 12+i*21))
        if policy_frame is not None:
            surface = pygame.surfarray.make_surface(np.transpose(policy_frame, (1, 0, 2)))
            self.window.blit(pygame.transform.scale(surface, (256, 240)), (792, 475))
        pygame.display.flip()
        return True

    def close(self):
        pygame.quit()
