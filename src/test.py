import pygame
import torch
import time
import random
import numpy as np
from pacman_env import PacmanEnv

# =================================================================
# [설정] 테스트할 모델 타입을 선택하세요.
# (train.py에서 학습시킨 모델과 같아야 파일을 찾을 수 있습니다)
MODEL_TYPE = "DDQN"
# =================================================================

# 파일 이름 자동 설정
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

    print(f"\n=== 🧠 {MODEL_TYPE} 모델 테스트 모드 ===")
    print(f"📂 불러올 파일: {model_filename}")

    try:
        # 모델 로드
        agent.model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
        agent.epsilon = 0.0 # 테스트니까 무조건 실력으로(Greedy)
        print(f">>> 로드 성공! AI가 플레이를 시작합니다.")
    except FileNotFoundError:
        print(f">>> 🚨 오류: '{model_filename}' 파일이 없습니다.")
        print(f">>> 먼저 train.py에서 MODEL_TYPE = '{MODEL_TYPE}'로 학습을 완료하세요.")
        return

    grid_state = env.reset()
    state = get_one_hot_state(grid_state)
    done = False
    total_reward = 0
    step = 0

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

        # 실시간 로그 출력
        print(f"Step: {step} | Reward: {reward:.2f} | Total: {total_reward:.2f}")

        env.render()
        time.sleep(0.05) # 속도 조절

    print(f"[{MODEL_TYPE}] 게임 종료! 최종 점수: {total_reward:.2f}, 생존: {step} 스텝")
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    run_test()