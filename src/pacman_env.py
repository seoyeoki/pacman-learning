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

        # 팩맨 위치 초기화
        pos = np.where(self.grid == PACMAN)
        self.pacman_pos = [pos[0][0], pos[1][0]]
        self.grid[self.pacman_pos[0], self.pacman_pos[1]] = EMPTY

        # 유령 위치 초기화
        g_idx = np.where(self.grid == GHOST)
        self.ghosts = [[g_idx[0][i], g_idx[1][i]] for i in range(len(g_idx[0]))]
        for gr, gc in self.ghosts:
            self.grid[gr, gc] = EMPTY

        # 코인 배치
        self.coins = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = COIN
                    self.coins.append([r, c])

        self.current_step = 0

        # [수정 1] self.state는 정의된 적이 없습니다. 현재 상태를 계산해서 반환해야 합니다.
        return self._get_observation()

    def step(self, action):
        """행동 수행 및 결과 반환"""
        self.current_step += 1
        reward = 0  # [방법 A 적용] 이동 감점 0 (숨만 쉬어도 나가는 돈 없음)
        done = False

        # 팩맨 이동 로직
        r, c = self.pacman_pos
        nr, nc = r + DX[action], c + DY[action]

        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and self.grid[nr, nc] != WALL:
            self.pacman_pos = [nr, nc]

            # [방법 A 적용] 코인 점수 +20 (이득!)
            if self.grid[nr, nc] == COIN:
                reward += 20
                self.grid[nr, nc] = EMPTY
                self.coins.remove([nr, nc])

                # 모든 코인 다 먹음
                if len(self.coins) == 0:
                    reward += 50  # 클리어 보너스
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

        # 유령 충돌 체크
        if self.pacman_pos in self.ghosts:
            reward = -100  # [방법 A 적용] 죽으면 큰 손해
            done = True

        # 최대 턴수 초과 체크
        if self.current_step >= self.max_steps:
            done = True

        # [수정 2] 다음 상태(next_state)를 계산해야 합니다.
        next_state = self._get_observation()

        # [수정 3] info 딕셔너리가 없으면 에러 날 수 있음
        info = {'step': self.current_step}

        # [수정 4] 정의된 변수들을 리턴합니다.
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