import pygame
import numpy as np
import random

# --- 상수 및 색상 정의 ---
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 600
GRID_SIZE = 20
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE
FPS = 60

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
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pacman RL Simulator")
        self.clock = pygame.time.Clock()

        # 맵 레이아웃 (1:벽, 0:길) - 20x20 크기에 맞게 나머지 공간은 벽으로 채워짐
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

        self.max_steps = 1000  # 강제 종료 리미트
        self.current_step = 0

        self.reset()

def reset(self):
    """게임 재시작 및 맵 초기화"""
    self.grid = self.base_grid.copy()
    # ... (기존 초기화 코드 유지) ...

    # [추가] 통계용 카운터 초기화
    self.wall_hits = 0
    self.coins_eaten = 0

    self.current_step = 0
    return self._get_observation()

def step(self, action):
    self.current_step += 1
    reward = -0.05  # [추천] 시간 페널티 강화 (-0.05 ~ -0.1)
    done = False

    r, c = self.pacman_pos
    nr, nc = r + DX[action], c + DY[action]

    # 벽 충돌 체크
    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE) or self.grid[nr, nc] == WALL:
        reward = -1.0
        self.wall_hits += 1  # [추가] 벽 충돌 카운트
    else:
        self.pacman_pos = [nr, nc]

        # 코인 먹기
        if self.grid[nr, nc] == COIN:
            reward += 10.0
            self.coins_eaten += 1  # [추가] 코인 섭취 카운트
            self.grid[nr, nc] = EMPTY
            self.coins.remove([nr, nc])

            if len(self.coins) == 0:
                reward += 50
                done = True

        # 3. 유령 이동
        for i, ghost in enumerate(self.ghosts):
            gr, gc = ghost
            moves = []

            # 유령도 상하좌우 중 벽이 아닌 곳을 찾음
            for d in range(4):
                ngr, ngc = gr + DX[d], gc + DY[d]
                if 0 <= ngr < GRID_SIZE and 0 <= ngc < GRID_SIZE and self.grid[ngr, ngc] != WALL:
                    moves.append([ngr, ngc])

            # 갈 곳이 있으면 랜덤하게 하나 골라서 이동
            if moves:
                self.ghosts[i] = random.choice(moves)

        # [추천 5] 사망 페널티: 회복 불가능한 수준 (-100)
        if self.pacman_pos in self.ghosts:
            reward = -100.0
            done = True

        # 시간 초과 (보통 점수 변화 없음 혹은 사망 처리)
        if self.current_step >= self.max_steps:
            done = True

        next_state = self._get_observation()

        # [수정] info 딕셔너리에 통계 정보 담아서 반환
        info = {
            'step': self.current_step,
            'wall_hits': self.wall_hits,
            'coins_eaten': self.coins_eaten
        }

        return next_state, reward, done, info

    def _get_observation(self):
        """20x20 그리드 상태 반환"""
        obs_grid = self.grid.copy()
        obs_grid[self.pacman_pos[0], self.pacman_pos[1]] = PACMAN
        for gr, gc in self.ghosts:
            obs_grid[gr, gc] = GHOST
        return obs_grid

    def render(self):
        """화면 그리기"""
        self.screen.fill(BLACK)
        current_view = self._get_observation()

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                val = current_view[r, c]
                rect = (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if val == WALL:
                    pygame.draw.rect(self.screen, BLUE, rect)
                elif val == COIN:
                    # 코인 그리기
                    pygame.draw.circle(self.screen, WHITE, (c*CELL_SIZE+CELL_SIZE//2, r*CELL_SIZE+CELL_SIZE//2), 4)
                elif val == PACMAN:
                    pygame.draw.circle(self.screen, YELLOW, (c*CELL_SIZE+CELL_SIZE//2, r*CELL_SIZE+CELL_SIZE//2), CELL_SIZE//2 - 2)
                elif val == GHOST:
                    pygame.draw.rect(self.screen, RED, (c*CELL_SIZE+4, r*CELL_SIZE+4, CELL_SIZE-8, CELL_SIZE-8))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()