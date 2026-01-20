import pygame
import torch
import time
import random
import numpy as np
from pacman_env import PacmanEnv
from dqn_agent import DQNAgent

# 학습 때와 똑같은 전처리 함수 필수!
def get_one_hot_state(grid):
    state_one_hot = np.zeros((5, 20, 20), dtype=np.float32)
    state_one_hot[0] = (grid == 0)
    state_one_hot[1] = (grid == 1)
    state_one_hot[2] = (grid == 2)
    state_one_hot[3] = (grid == 3)
    state_one_hot[4] = (grid == 4)
    return state_one_hot.flatten()

def run_test(mode='trained'):
    env = PacmanEnv()

    # 모델 크기 맞춰주기
    state_size = 20 * 20 * 5
    action_size = 4

    agent = DQNAgent(state_size, action_size)

    if mode == 'trained':
        print("\n=== 🧠 학습된 AI(After) 로딩 중... ===")
        try:
            agent.model.load_state_dict(torch.load("pacman_dqn.pth", map_location=torch.device('cpu')))
            agent.epsilon = 0.0
            print(">>> 모델 로드 성공! AI가 플레이합니다.")
        except FileNotFoundError:
            print(">>> 🚨 'pacman_dqn.pth' 파일이 없습니다. 먼저 학습을 돌려주세요.")
            return
    else:
        print("\n=== 🎲 랜덤 팩맨(Before) 시작... ===")

    grid_state = env.reset()
    state = get_one_hot_state(grid_state) # [변경] 원-핫
    done = False
    total_reward = 0
    step = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        if mode == 'random':
            action = random.choice([0, 1, 2, 3])
        else:
            action = agent.get_action(state)

        next_grid_state, reward, done, _ = env.step(action)
        state = get_one_hot_state(next_grid_state) # [변경] 원-핫
        total_reward += reward
        step += 1

        env.render()
        time.sleep(0.05) # 관전하기 좋은 속도

    print(f"[{mode.upper()}] 게임 종료! 점수: {total_reward:.2f}, 생존: {step}")
    time.sleep(1)
    env.close()

if __name__ == "__main__":
    # 학습된 모델 테스트
    run_test(mode='trained')