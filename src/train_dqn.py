import pygame
import numpy as np
import random

# --- 상수 및 색상 정의 ---
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 600
GRID_SIZE = 20
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE
FPS = 60 # 학습 속도를 위해 프레임 제한을 조금 높였습니다

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
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DX = [-1, 0, 1, 0]
DY = [0, 1, 0, -1]

class PacmanEnv:
    def __init__(self):
        # [수정됨] 시작하자마자 PyGame 시스템과 화면을 초기화합니다.
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pacman RL Simulator")
        self.clock = pygame.time.Clock()

        # 맵 레이아웃 (1:벽, 0:길)
        self.map_layout = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

        self.base_grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=int) * WALL
        r, c = self.map_layout.shape
        self.base_grid[:r, :c] = self.map_layout

        self.reset()

    def reset(self):
        """게임 재시작 및 맵 초기화"""
        self.grid = self.base_grid.copy()

        pos = np.where(self.grid == PACMAN)
        self.pacman_pos = [pos[0][0], pos[1][0]]
        self.grid[self.pacman_pos[0], self.pacman_pos[1]] = EMPTY

        g_idx = np.where(self.grid == GHOST)
        self.ghosts = [[g_idx[0][i], g_idx[1][i]] for i in range(len(g_idx[0]))]
        for gr, gc in self.ghosts:
            self.grid[gr, gc] = EMPTY

        self.coins = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = COIN
                    self.coins.append([r, c])

        self.step_count = 0
        return self._get_observation()

    def step(self, action):
        """행동 수행 및 결과 반환"""
        self.step_count += 1
        reward = -1  # 시간 페널티
        done = False

        # 팩맨 이동
        r, c = self.pacman_pos
        nr, nc = r + DX[action], c + DY[action]

        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and self.grid[nr, nc] != WALL:
            self.pacman_pos = [nr, nc]
            if self.grid[nr, nc] == COIN:
                reward += 10
                self.grid[nr, nc] = EMPTY
                self.coins.remove([nr, nc])
                if len(self.coins) == 0:
                    reward += 50
                    done = True

        # 유령 이동 (랜덤)
        for i, ghost in enumerate(self.ghosts):
            gr, gc = ghost
            moves = []
            for d in range(4):
                ngr, ngc = gr + DX[d], gc + DY[d]
                if 0 <= ngr < GRID_SIZE and 0 <= ngc < GRID_SIZE and self.grid[ngr, ngc] != WALL:
                    moves.append([ngr, ngc])
            if moves:
                self.ghosts[i] = random.choice(moves)

        if self.pacman_pos in self.ghosts:
            reward = -100
            done = True

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """20x20 그리드 상태 반환"""
        obs_grid = self.grid.copy()
        obs_grid[self.pacman_pos[0], self.pacman_pos[1]] = PACMAN
        for gr, gc in self.ghosts:
            obs_grid[gr, gc] = GHOST
        return obs_grid

    def render(self):
        """화면 그리기"""
        # 초기화 코드는 __init__으로 이동했으므로 여기선 그리기만 합니다.
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
        # FPS 조절은 렌더링 할 때만 합니다.
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()