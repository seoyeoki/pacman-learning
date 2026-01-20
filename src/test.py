import pygame
import torch
import time
import random
import numpy as np
from pacman_env import PacmanEnv

# =================================================================
# [설정] 테스트할 모델을 문자열로 지정하세요.
MODEL_TYPE = "DDQN"  # "DQN", "DDQN", "DUELING"
# =================================================================

# 파일명 자동 설정
model_filename = f"pacman_{MODEL_TYPE.lower()}.pth"

if MODEL_TYPE == "DQN":
    from dqn_agent import DQNAgent as Agent
elif MODEL_TYPE == "DDQN":
    from ddqn_agent import DDQNAgent as Agent
elif MODEL_TYPE == "DUELING":
    from dueling_agent import DuelingAgent as Agent
else:
    raise ValueError(f"Unknown Model Type: {MODEL_TYPE}")


def get_one_hot_state(grid):
    state_one_hot = np.zeros((5, 20, 20), dtype=np.float32)
    state_one_hot[0] = (grid == 0)
    state_one_hot[1] = (grid == 1)
    state_one_hot[2] = (grid == 2)
    state_one_hot[3] = (grid == 3)
    state_one_hot[4] = (grid == 4)
    return state_one_hot.flatten()

def run_test():
    env = PacmanEnv()
    state_size = 20 * 20 * 5
    action_size = 4
    agent = Agent(state_size, action_size)

    print(f"\n=== 🧠 {MODEL_TYPE} 모델 로딩 중... ===")
    print(f"Target File: {model_filename}")

    try:
        agent.model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
        agent.epsilon = 0.0
        print(f">>> 로드 성공! AI가 플레이합니다.")
    except FileNotFoundError:
        print(f">>> 🚨 파일이 없습니다. 먼저 '{MODEL_TYPE}' 모드로 학습을 돌려주세요.")
        return

    grid_state = env.reset()
    state = get_one_hot_state(grid_state)
    done = False
    total_reward = 0
    step = 0

    print(f"--- {MODEL_TYPE} Play Start ---")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        action = agent.get_action(state)
        next_grid_state, reward, done, info = env.step(action)
        state = get_one_hot_state(next_grid_state)

        total_reward += reward
        step += 1

        print(f"Step: {step} | Reward: {reward:.2f} | Total: {total_reward:.2f}")

        env.render()
        time.sleep(0.05)

    print(f"[{MODEL_TYPE}] 종료! 점수: {total_reward:.2f}, 스텝: {step}")
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    run_test()