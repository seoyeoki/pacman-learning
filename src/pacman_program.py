import pygame
import torch
import numpy as np
import sys
import json
import time

# 기존 환경 및 에이전트 import
from pacman_env import PacmanEnv, CELL_SIZE, GRID_SIZE
from pacman_env import EMPTY, WALL, PACMAN, GHOST, COIN
from cnn_model_agent.cnn_ddqn_agent import CNNDDQNAgent
from cnn_model_agent.random_agent import RandomAgent

# ==========================================
# [설정] 색상 및 UI 상수
# ==========================================
GRAY = (50, 50, 50)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

FPS = 10
TOP_MARGIN = 50   # 상단 점수판 높이
BOTTOM_MARGIN = 80 # 하단 설명창 높이

class PacmanDemo:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()

        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_desc = pygame.font.SysFont("Arial", 16)

        # 1. 환경 및 에이전트 로드
        self.env = PacmanEnv()
        self.agent_ddqn = CNNDDQNAgent(action_size=4)
        self.agent_random = RandomAgent(action_size=4)

        # 모델 로드
        try:
            # 경로가 맞는지 확인 필요 (상황에 따라 ../ 또는 ./ 조정)
            self.agent_ddqn.model.load_state_dict(
                torch.load("../trained_pth/pacman_cnn_ddqn.pth", map_location='cpu', weights_only=True)
            )
            self.agent_ddqn.epsilon = 0.05
            self.agent_ddqn.model.eval()
            print("✅ DDQN 모델 로드 완료")
        except:
            print("⚠️ 모델 파일 없음. 랜덤 에이전트로 대체.")

        # 2. 상태 변수
        self.current_agent = "DDQN"
        self.mode = "MENU"

        # 통계 변수
        self.step_count = 0
        self.total_score = 0
        self.coins_eaten = 0

        # v3 유령 제어 변수
        self.scripted_ghosts = []
        self.is_dragging = False
        self.current_drag_path = []
        self.dragging_ghost_idx = None

        # 코인 그리기 모드 (스페이스바)
        self.painting_coin = False

        # 창 크기 초기 설정
        self.update_window_size()

    def update_window_size(self):
        """현재 맵 크기에 맞춰 창 크기 재조정"""
        current_rows, current_cols = self.env.grid.shape
        self.map_width = current_cols * CELL_SIZE
        self.map_height = current_rows * CELL_SIZE
        self.window_width = self.map_width
        self.window_height = self.map_height + TOP_MARGIN + BOTTOM_MARGIN

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(f"Pacman AI Demo (Map: {current_cols}x{current_rows})")

    def reset_stats(self):
        self.step_count = 0
        self.total_score = 0
        self.coins_eaten = 0

    def reset_v1(self):
        self.env.reset()
        self.reset_stats()
        self.update_window_size()
        self.mode = "PLAY"

    def reset_v2(self):
        self.env.reset()
        rows, cols = self.env.grid.shape
        for r in range(rows):
            for c in range(cols):
                if self.env.grid[r, c] == COIN:
                    self.env.grid[r, c] = EMPTY
        self.env.coins = []
        self.reset_stats()
        self.update_window_size()
        self.mode = "EDIT_V2"

    def reset_v3(self):
        """저장된 맵 로드"""
        try:
            with open("custom_map.json", "r") as f:
                data = json.load(f)
                self.env.grid = np.array(data["grid"])
                self.env.pacman_pos = data["pacman"]
                self.env.ghosts = []
                self.env.coins = []
                self.update_env_coins()
                self.update_window_size()
        except FileNotFoundError:
            print("❌ custom_map.json 없음!")
            self.env.reset()
            self.update_window_size()

        self.reset_stats()
        self.scripted_ghosts = []
        self.mode = "EDIT_V3_GHOSTS"

    def get_state(self):
        rows, cols = self.env.grid.shape
        grid = self.env.grid
        pacman_pos = self.env.pacman_pos
        ghosts = self.env.ghosts

        state = np.zeros((5, rows, cols), dtype=np.float32)
        state[0] = (grid == 0)
        state[1] = (grid == 1)
        state[4] = (grid == 4)
        state[2][pacman_pos[0], pacman_pos[1]] = 1.0
        for gr, gc in ghosts:
            if 0 <= gr < rows and 0 <= gc < cols:
                state[3][gr, gc] = 1.0
        return state

    # --- 헬퍼 함수들 ---
    def get_mouse_grid_pos(self):
        mx, my = pygame.mouse.get_pos()
        adj_y = my - TOP_MARGIN
        return adj_y // CELL_SIZE, mx // CELL_SIZE

    def is_valid_pos(self, r, c):
        rows, cols = self.env.grid.shape
        return 0 <= r < rows and 0 <= c < cols

    def update_env_coins(self):
        self.env.coins = []
        rows, cols = self.env.grid.shape
        for r in range(rows):
            for c in range(cols):
                if self.env.grid[r, c] == COIN:
                    self.env.coins.append([r,c])

    def toggle_coin_at_mouse(self):
        r, c = self.get_mouse_grid_pos()
        if self.is_valid_pos(r, c) and 0 < r < self.env.grid.shape[0]-1 and 0 < c < self.env.grid.shape[1]-1:
            if self.env.grid[r, c] == COIN: self.env.grid[r, c] = EMPTY
            elif self.env.grid[r, c] == EMPTY: self.env.grid[r, c] = COIN

    def set_coin_at_mouse(self):
        """스페이스바 드래그용 (강제 코인 생성)"""
        r, c = self.get_mouse_grid_pos()
        if self.is_valid_pos(r, c) and 0 < r < self.env.grid.shape[0]-1 and 0 < c < self.env.grid.shape[1]-1:
            if self.env.grid[r, c] == EMPTY:
                self.env.grid[r, c] = COIN

    def find_ghost_index_at(self, r, c):
        for i, ghost in enumerate(self.scripted_ghosts):
            curr_pos = ghost['path'][ghost['idx']]
            if curr_pos == [r, c]:
                return i
        return None

    def record_drag_path(self):
        r, c = self.get_mouse_grid_pos()
        if self.is_valid_pos(r, c):
            last_r, last_c = self.current_drag_path[-1]
            if abs(r - last_r) + abs(c - last_c) == 1:
                if self.env.grid[r, c] != WALL:
                    self.current_drag_path.append([r, c])

    def finish_drag_new_ghost(self):
        self.is_dragging = False
        # 클릭만 해도 생성되도록 len > 0으로 설정
        if len(self.current_drag_path) > 0:
            self.scripted_ghosts.append({
                'path': list(self.current_drag_path),
                'idx': 0
            })
        self.current_drag_path = []

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False

            if self.mode == "MENU":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1: self.reset_v1()
                    elif event.key == pygame.K_2: self.reset_v2()
                    elif event.key == pygame.K_3: self.reset_v3()
                    elif event.key == pygame.K_a:
                        self.current_agent = "RANDOM" if self.current_agent == "DDQN" else "DDQN"

            elif self.mode == "EDIT_V2":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    self.mode = "PLAY"
                    self.update_env_coins()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.toggle_coin_at_mouse()

            elif self.mode == "EDIT_V3_GHOSTS":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.mode = "PLAY_V3"
                        self.env.ghosts = [g['path'][0] for g in self.scripted_ghosts]
                        self.update_env_coins()
                    if event.key == pygame.K_c:
                        self.scripted_ghosts = []
                        self.env.ghosts = []
                    if event.key == pygame.K_SPACE:
                        self.painting_coin = True

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        self.painting_coin = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    # [좌클릭] 유령 생성/드래그
                    if event.button == 1:
                        self.is_dragging = True
                        r, c = self.get_mouse_grid_pos()
                        if self.is_valid_pos(r, c):
                            self.current_drag_path = [[r, c]]

                    # [휠 클릭] 코인 토글
                    elif event.button == 2:
                        self.toggle_coin_at_mouse()

                    # [우클릭] 팩맨 위치 지정
                    elif event.button == 3:
                        r, c = self.get_mouse_grid_pos()
                        if self.is_valid_pos(r, c) and self.env.grid[r, c] != WALL:
                            self.env.pacman_pos = [r, c]

                elif event.type == pygame.MOUSEMOTION:
                    if self.is_dragging:
                        self.record_drag_path()
                    if self.painting_coin:
                        self.set_coin_at_mouse()

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.finish_drag_new_ghost()

            elif "PLAY" in self.mode and self.mode == "PLAY_V3":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # 좌클릭으로 유령 납치
                        r, c = self.get_mouse_grid_pos()
                        found_idx = self.find_ghost_index_at(r, c)
                        if found_idx is not None:
                            self.dragging_ghost_idx = found_idx
                            self.current_drag_path = [[r, c]]

                elif event.type == pygame.MOUSEMOTION and self.dragging_ghost_idx is not None:
                    self.record_drag_path()

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        if self.dragging_ghost_idx is not None and len(self.current_drag_path) > 0:
                            target_ghost = self.scripted_ghosts[self.dragging_ghost_idx]
                            target_ghost['path'] = list(self.current_drag_path)
                            target_ghost['idx'] = 0
                        self.dragging_ghost_idx = None
                        self.current_drag_path = []

        return True

    def move_scripted_ghosts(self):
        new_positions = []
        for i, ghost in enumerate(self.scripted_ghosts):
            if i == self.dragging_ghost_idx:
                new_positions.append(ghost['path'][ghost['idx']])
                continue

            path = ghost['path']
            idx = ghost['idx']
            if idx < len(path) - 1:
                idx += 1
                ghost['idx'] = idx
            new_positions.append(path[idx])

        self.env.ghosts = new_positions

    def run(self):
        running = True
        while running:
            running = self.handle_input()

            self.screen.fill(GRAY)

            # 상단 UI
            pygame.draw.rect(self.screen, BLACK, (0, 0, self.window_width, TOP_MARGIN))
            info_str = f"STEP: {self.step_count} | COINS: {self.coins_eaten} | SCORE: {self.total_score:.1f}"
            info_surf = self.font_ui.render(info_str, True, GREEN)
            text_rect = info_surf.get_rect(center=(self.window_width // 2, TOP_MARGIN // 2))
            self.screen.blit(info_surf, text_rect)

            # 맵 그리기
            pygame.draw.rect(self.screen, BLACK, (0, TOP_MARGIN, self.window_width, self.map_height))
            rows, cols = self.env.grid.shape
            for r in range(rows):
                for c in range(cols):
                    rect = (c*CELL_SIZE, r*CELL_SIZE + TOP_MARGIN, CELL_SIZE, CELL_SIZE)
                    val = self.env.grid[r, c]
                    if val == WALL: pygame.draw.rect(self.screen, BLUE, rect)
                    elif val == COIN: pygame.draw.circle(self.screen, WHITE, (rect[0]+15, rect[1]+15), 4)

            pr, pc = self.env.pacman_pos
            pygame.draw.circle(self.screen, YELLOW, (pc*CELL_SIZE+15, pr*CELL_SIZE+15 + TOP_MARGIN), 13)

            # 유령 및 경로
            if self.mode == "EDIT_V3_GHOSTS" or (self.mode == "PLAY_V3" and len(self.current_drag_path) > 0):
                if len(self.current_drag_path) > 1:
                    pts = [(c*CELL_SIZE+10, r*CELL_SIZE+10+TOP_MARGIN) for r, c in self.current_drag_path]
                    pygame.draw.lines(self.screen, RED, False, pts, 3)

            if self.mode == "EDIT_V3_GHOSTS":
                for g in self.scripted_ghosts:
                    path = g['path']
                    if len(path) > 1:
                        pts = [(c*CELL_SIZE+10, r*CELL_SIZE+10+TOP_MARGIN) for r, c in path]
                        pygame.draw.lines(self.screen, (200, 100, 100), False, pts, 2)

                    gr, gc = path[0]
                    pygame.draw.rect(self.screen, RED, (gc*CELL_SIZE+5, gr*CELL_SIZE+5 + TOP_MARGIN, 20, 20))
            else:
                for gr, gc in self.env.ghosts:
                    pygame.draw.rect(self.screen, RED, (gc*CELL_SIZE+5, gr*CELL_SIZE+5 + TOP_MARGIN, 20, 20))

            # 게임 로직
            if "PLAY" in self.mode:
                state = self.get_state()

                # ========================================================
                # [수정] 한 줄 코드를 안전한 if-else 블록으로 변경 (SyntaxError 해결)
                # ========================================================
                if self.current_agent == "DDQN":
                    action = self.agent_ddqn.get_action(state)
                else:
                    action = self.agent_random.get_action(state)

                if self.mode == "PLAY_V3":
                    _, reward, done, info = self.env.step(action)
                    self.move_scripted_ghosts()
                else:
                    _, reward, done, info = self.env.step(action)

                self.step_count += 1
                self.total_score += reward
                self.coins_eaten = info.get('coins_eaten', 0)

                if done:
                    print("게임 종료! 메뉴로 복귀")
                    time.sleep(1)
                    self.mode = "MENU"

            # 하단 설명창
            desc = ""
            if self.mode == "MENU": desc = "[1] Random  [2] Coin Edit  [3] Custom Map"
            elif self.mode == "EDIT_V2": desc = "[Click] Coin Toggle  [Enter] Start"
            elif self.mode == "EDIT_V3_GHOSTS":
                desc = "[L-Click/Drag] Ghost  [R-Click] Pacman  [Mid-Click/Space] Coin  [Enter] Start"
            elif self.mode == "PLAY_V3": desc = "[Drag Ghost] Re-route Path!"

            desc_surf = self.font_desc.render(desc, True, WHITE)
            self.screen.blit(desc_surf, (20, self.window_height - BOTTOM_MARGIN + 20))

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    PacmanDemo().run()