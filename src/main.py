# main.py 라고 가정
from pacman_env import PacmanEnv
import random
import time

# --- 1. 랜덤 모델 (바보 에이전트) ---
def random_model(observation):
    return random.choice([0, 1, 2, 3])

# --- 2. 규칙 기반 모델 (하드코딩된 지능) ---
def rule_based_model(observation):
    # observation이 [상, 우, 하, 좌] 센서값이라고 가정 (mode='sensor')
    # 예: 위에 유령이 있으면(1), 아래(2)로 도망가라
    if observation[0] == 1: return 2
    if observation[1] == 1: return 3
    if observation[2] == 1: return 0
    if observation[3] == 1: return 1
    return random.choice([0, 1, 2, 3]) # 위험 없으면 랜덤

# --- 3. 실행기 (Controller) ---
def run_simulation(model_func, mode='sensor'):
    env = PacmanEnv(mode=mode)
    obs = env.reset()
    total_reward = 0

    print(f"--- Simulation Start with {model_func.__name__} ---")

    running = True
    while running:
        # [핵심] 시뮬레이터가 준 데이터(obs)를 모델에 넣고, 행동(action)을 받음
        action = model_func(obs)

        # 행동을 시뮬레이터에 입력
        obs, reward, done, info = env.step(action)
        total_reward += reward

        env.render()

        if done:
            print(f"Game Over! Total Score: {total_reward}")
            running = False
            time.sleep(1) # 결과 확인 대기

    env.close()

# --- 실행 ---
if __name__ == "__main__":
    # 1. 랜덤 모델 테스트
    run_simulation(random_model, mode='sensor')

    # 2. 규칙 기반 모델 테스트 (조금 더 오래 살 것입니다)
    # run_simulation(rule_based_model, mode='sensor')