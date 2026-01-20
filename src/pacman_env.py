import pygame
import numpy as np
import random

# --- 상수 및 색상 정의 ---
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 600
GRID_SIZE = 20
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE
FPS = 60  # 학습 속도가 너무 빠르면 30으로 낮추거나, render를 끄세요.

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)      # 벽
YELLOW = (255, 255, 0)  # 팩맨
RED = (255, 0, 0)       # 유령
WHITE = (255, 255, 255) # 코인

# 그리드 객체 코드
EMPTY = 0
WALL = 1
PACMAN = 2
GHOST = 3
COIN = 4

# 행동 정의
DX = [-1, 0, 1, 0]
DY = [0, 1, 0, -1]

class PacmanEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pacman RL Simulator (New Map)")
        self.clock = pygame.time.Clock()

        # [NEW] 죽은 공간 없는 20x20 풀 사이즈 미로 맵
        # 1: 벽, 0: 길 (초기 설정)
        self.map_layout = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

        self.base_grid = self.map_layout.copy()
        self.max_steps = 1000
        self.current_step = 0

        self.reset()

    def reset(self):
        """게임 재시작"""
        self.grid = self.base_grid.copy()

        # 1. 코인 배치 (빈 길에 모두 배치)
        self.coins = []
        empty_spots = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.grid[r, c] == EMPTY:
                    self.grid[r, c] = COIN
                    self.coins.append([r, c])
                    empty_spots.append([r, c])

        # 2. 팩맨 위치 (빈 곳 중 랜덤)
        start_pos = random.choice(empty_spots)
        self.pacman_pos = [start_pos[0], start_pos[1]]
        self.grid[self.pacman_pos[0], self.pacman_pos[1]] = EMPTY # 시작 위치 코인은 제거 (선택사항)
        if [start_pos[0], start_pos[1]] in self.coins:
            self.coins.remove([start_pos[0], start_pos[1]])

        # 3. 유령 위치 (빈 곳 중 랜덤 2마리)
        self.ghosts = []
        for _ in range(2):
            while True:
                g_pos = random.choice(empty_spots)
                # 팩맨과 너무 가까우면 다시 뽑기
                if abs(g_pos[0] - self.pacman_pos[0]) + abs(g_pos[1] - self.pacman_pos[1]) > 5:
                    self.ghosts.append(g_pos)
                    self.grid[g_pos[0], g_pos[1]] = EMPTY
                    break

        # 통계 초기화
        self.wall_hits = 0
        self.coins_eaten = 0
        self.current_step = 0

        return self._get_observation()

    def step(self, action):
        self.current_step += 1

        # [중요] 페널티 설정 (-0.1: 움직여라!)
        reward = -0.1
        done = False

        r, c = self.pacman_pos
        nr, nc = r + DX[action], c + DY[action]

        # 벽 충돌 체크
        if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE) or self.grid[nr, nc] == WALL:
            reward = -1.0
            self.wall_hits += 1
        else:
            self.pacman_pos = [nr, nc]

            # 코인 획득
            if self.grid[nr, nc] == COIN:
                reward += 10.0
                self.coins_eaten += 1
                self.grid[nr, nc] = EMPTY
                if [nr, nc] in self.coins:
                    self.coins.remove([nr, nc])

                if len(self.coins) == 0:
                    reward += 50 # 클리어
                    done = True

        # 유령 처리
        for i, ghost in enumerate(self.ghosts):
            gr, gc = ghost
            is_coin = [gr, gc] in self.coins
            self.grid[gr, gc] = COIN if is_coin else EMPTY

            moves = []
            for d in range(4):
                ngr, ngc = gr + DX[d], gc + DY[d]
                if 0 <= ngr < GRID_SIZE and 0 <= ngc < GRID_SIZE and self.grid[ngr, ngc] != WALL:
                    moves.append([ngr, ngc])

            if moves:
                self.ghosts[i] = random.choice(moves)
                # 유령 시각화는 render나 observation에서 처리

        # 유령 충돌 (사망)
        if self.pacman_pos in self.ghosts:
            reward = -100.0
            done = True

        if self.current_step >= self.max_steps:
            done = True

        info = {
            'step': self.current_step,
            'wall_hits': self.wall_hits,
            'coins_eaten': self.coins_eaten
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        obs_grid = self.grid.copy()
        obs_grid[self.pacman_pos[0], self.pacman_pos[1]] = PACMAN
        for gr, gc in self.ghosts:
            obs_grid[gr, gc] = GHOST
        return obs_grid

    def render(self):
        self.screen.fill(BLACK)
        current_view = self._get_observation()

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                val = current_view[r, c]
                rect = (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)

                if val == WALL:
                    pygame.draw.rect(self.screen, BLUE, rect)
                elif val == COIN:
                    pygame.draw.circle(self.screen, WHITE, (c*CELL_SIZE+CELL_SIZE//2, r*CELL_SIZE+CELL_SIZE//2), 4)
                elif val == PACMAN:
                    pygame.draw.circle(self.screen, YELLOW, (c*CELL_SIZE+CELL_SIZE//2, r*CELL_SIZE+CELL_SIZE//2), CELL_SIZE//2 - 2)
                elif val == GHOST:
                    pygame.draw.rect(self.screen, RED, (c*CELL_SIZE+4, r*CELL_SIZE+4, CELL_SIZE-8, CELL_SIZE-8))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()