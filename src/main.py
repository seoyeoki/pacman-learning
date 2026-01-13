import sys
import random
import time
from pacman_env import PacmanEnv  # 우리가 만든 파일 불러오기

def run_game():
    # 1. 환경 생성
    env = PacmanEnv()

    # 2. 에피소드 반복 (학습 Loop)
    episodes = 5

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0

        print(f"--- Episode {episode} Start ---")

        while not done:
            # ----------------------------------------
            # [Model Area] 나중에 여기에 AI 모델이 들어갑니다.
            # 지금은 랜덤으로 아무 방향이나 선택합니다.
            action = random.choice([0, 1, 2, 3])
            # ----------------------------------------

            # 환경에 행동 입력
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state

            # 화면 표시
            env.render()

            # 게임 종료 이벤트 처리 (창 닫기 등)
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    sys.exit()

        print(f"Episode {episode} Finished. Total Score: {total_reward}")
        time.sleep(0.5) # 결과 확인용 잠시 대기

    env.close()

if __name__ == "__main__":
    run_game()