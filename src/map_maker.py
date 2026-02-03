import pygame
import json
import numpy as np
import sys
from pacman_env import SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE, GRID_SIZE, WALL, EMPTY, COIN, PACMAN

# 색상 설정
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GRAY = (50, 50, 50)

class MapMaker:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 60))
        pygame.display.set_caption("Pacman Map Maker (Only Walls)")
        self.clock = pygame.time.Clock()

        # 맵 데이터 초기화 (테두리는 벽으로 고정)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

        self.tool = WALL # 기본 도구: 벽

        # 호환성을 위해 팩맨 위치는 (1,1)로 고정 저장
        self.pacman_pos = [1, 1]

    def save_map(self, filename="custom_map.json"):
        # pacman_program.py에서 읽을 수 있도록 형식 유지
        data = {
            "grid": self.grid.tolist(),
            "pacman": self.pacman_pos
        }
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"💾 맵 저장 완료: {filename}")

    def load_map(self, filename="custom_map.json"):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                self.grid = np.array(data["grid"])
                # 불러온 맵에 팩맨 정보가 있으면 가져오고, 없으면 기본값
                self.pacman_pos = data.get("pacman", [1, 1])
            print(f"📂 맵 불러오기 완료: {filename}")
        except FileNotFoundError:
            print("⚠️ 저장된 맵 파일이 없습니다.")

    def run(self):
        running = True
        font = pygame.font.SysFont("Arial", 20)

        while running:
            self.screen.fill(GRAY)

            # --- 입력 처리 ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

                # 키보드 입력
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1: self.tool = WALL   # 벽 그리기
                    if event.key == pygame.K_2: self.tool = EMPTY  # 지우개 (빈 공간)
                    if event.key == pygame.K_s: self.save_map()
                    if event.key == pygame.K_l: self.load_map()

                # 마우스 입력 (그리기)
                if pygame.mouse.get_pressed()[0]: # 좌클릭 유지
                    mx, my = pygame.mouse.get_pos()
                    c, r = mx // CELL_SIZE, my // CELL_SIZE

                    # 테두리는 수정 불가 (0 < r, c < GRID_SIZE-1)
                    if 0 < r < GRID_SIZE-1 and 0 < c < GRID_SIZE-1:
                        self.grid[r, c] = self.tool

            # --- 렌더링 ---
            # 맵 영역 배경 (검정)
            pygame.draw.rect(self.screen, BLACK, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    rect = (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    if self.grid[r, c] == WALL:
                        pygame.draw.rect(self.screen, BLUE, rect)

            # UI 텍스트
            tool_name = "WALL" if self.tool == WALL else "ERASER"
            ui_text = f"Tool: {tool_name} (1:Wall, 2:Eraser) | [S]ave | [L]oad"
            self.screen.blit(font.render(ui_text, True, WHITE), (10, SCREEN_HEIGHT + 15))

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()

if __name__ == "__main__":
    MapMaker().run()