import pygame
import numpy as np
import random
import sys

# --- 상수 정의 ---
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 600
GRID_SIZE = 20
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE
FPS = 20  # 학습 속도를 위해 약간 빠르게

BLACK, BLUE, YELLOW, RED, GREEN = (0,0,0), (0,0,255), (255,255,0), (255,0,0), (0,255,0)
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DX = [-1, 0, 1, 0]
DY = [0, 1, 0, -1]

class PacmanEnv:
    def __init__(self, mode='sensor'):
        """
        mode:
          - 'sensor': 논문 방식 (4방향 유령 센서 값만 반환)
          - 'grid': 전체 맵 그리드 반환 (추후 DQN 등 확장용)
        """
        self.mode = mode
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f"Pacman Simulator - Mode: {self.mode}")
        self.clock = pygame.time.Clock()

        # 맵: 1=벽, 0=빈공간, 2=팩맨, 3=유령
        self.map_layout = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

        self.grid_base = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)
        r, c = self.map_layout.shape
        self.grid_base[:r, :c] = self.map_layout
        self.grid = self.grid_base.copy()

    def reset(self):
        """새로운 에피소드 시작"""
        self.grid = self.grid_base.copy()

        pos = np.where(self.grid == 2)
        self.pacman_pos = [pos[0][0], pos[1][0]]

        g_idx = np.where(self.grid == 3)
        self.ghosts = [[g_idx[0][i], g_idx[1][i]] for i in range(len(g_idx[0]))]

        return self._get_observation()

    def step(self, action):
        """행동 수행 -> (다음 상태, 보상, 종료 여부, 정보) 반환"""
        # 1. 팩맨 이동
        r, c = self.pacman_pos
        nr, nc = r + DX[action], c + DY[action]
        if self.grid[nr, nc] != 1:
            self.pacman_pos = [nr, nc]

        # 2. 유령 이동 (랜덤)
        for i, ghost in enumerate(self.ghosts):
            gr, gc = ghost
            moves = []
            for d in range(4):
                ngr, ngc = gr + DX[d], gc + DY[d]
                if self.grid[ngr, ngc] != 1:
                    moves.append([ngr, ngc])
            if moves:
                self.ghosts[i] = random.choice(moves)

        # 3. 보상 및 종료 조건
        reward = 1  # 생존 보상
        done = False

        if self.pacman_pos in self.ghosts:
            done = True
            reward = -100 # 사망 페널티

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """설정된 모드에 따라 다른 형태의 데이터를 반환"""
        if self.mode == 'sensor':
            # 논문 방식: 4방향 센서값 (유령 감지 여부)
            sensors = [0, 0, 0, 0] # 상, 우, 하, 좌
            pr, pc = self.pacman_pos
            for d in range(4):
                dist = 1
                while True:
                    nr, nc = pr + DX[d] * dist, pc + DY[d] * dist
                    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE or self.grid[nr, nc] == 1:
                        break
                    if [nr, nc] in self.ghosts:
                        sensors[d] = 1
                        break # 가장 가까운 유령 하나만 감지
                    dist += 1
            return np.array(sensors)

        elif self.mode == 'grid':
            # 딥러닝 방식: 현재 맵 전체 상태 반환 (20x20 행렬)
            # (시각화를 위해 원본 grid에 팩맨/유령 위치를 업데이트한 복사본 생성)
            current_grid = self.grid.copy()
            # 팩맨과 유령은 움직이므로 grid 값 갱신 필요 (여기선 단순화)
            # 실제 학습용으로는 Channel 분리 등이 필요할 수 있음
            return current_grid

    def render(self):
        """화면 출력"""
        self.screen.fill(BLACK)
        # 벽
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.grid[r, c] == 1:
                    pygame.draw.rect(self.screen, BLUE, (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # 팩맨
        pr, pc = self.pacman_pos
        pygame.draw.circle(self.screen, YELLOW, (pc*CELL_SIZE+15, pr*CELL_SIZE+15), 10)

        # 유령
        for gr, gc in self.ghosts:
            pygame.draw.rect(self.screen, RED, (gc*CELL_SIZE+5, gr*CELL_SIZE+5, 20, 20))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()