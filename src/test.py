import pygame
import torch
import time
import numpy as np
import csv
from datetime import datetime
from pacman_env import PacmanEnv

# =================================================================
# [설정] 테스트할 모델 타입을 선택하세요.
# -----------------------------------------------------------------
# 1. "DQN"     : 기본 DQN (Target Network만 사용)
# 2. "DDQN"    : Double DQN (과대평가 방지)
# 3. "DUELING" : Dueling DQN (상태 가치와 행동 이점 분리)
# 4. "RANDOM"  : 학습 없는 무작위 행동 (비교용)
# -----------------------------------------------------------------
MODEL_TYPE = "DDQN"
# =================================================================

# 1. 현재 시간 구하기 (파일명용)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 2. 결과 파일명 생성 (매번 새로운 파일)
RESULT_FILENAME = f"test_result_{MODEL_TYPE}_{current_time}.csv"


# --- 모델 타입에 따른 클래스 및 파일 설정 ---
if MODEL_TYPE == "RANDOM":
    from src.model_agent.random_agent import RandomAgent as AgentClass
    model_filename = None

elif MODEL_TYPE == "DQN":
    from src.model_agent.dqn_agent import DQNAgent as AgentClass
    model_filename = "../trained_pth/pacman_dqn.pth"

elif MODEL_TYPE == "DDQN":
    from src.model_agent.ddqn_agent import DDQNAgent as AgentClass
    model_filename = "../trained_pth/pacman_ddqn.pth"

elif MODEL_TYPE == "DUELING":
    from dueling_agent import DuelingAgent as AgentClass
    model_filename = "pacman_dueling.pth"

else:
    raise ValueError(f"지원하지 않는 모델 타입입니다: {MODEL_TYPE}")


def get_one_hot_state(grid):
    state_one_hot = np.zeros((5, 20, 20), dtype=np.float32)
    state_one_hot[0] = (grid == 0)
    state_one_hot[1] = (grid == 1)
    state_one_hot[2] = (grid == 2)
    state_one_hot[3] = (grid == 3)
    state_one_hot[4] = (grid == 4)
    return state_one_hot.flatten()

def save_result_to_csv(model_type, score, steps, wall_hits, coins):
    """결과 저장 함수"""
    with open(RESULT_FILENAME, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Model_Type', 'Score', 'Steps', 'Wall_Hits', 'Coins'])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, model_type, score, steps, wall_hits, coins])

    print(f"💾 성적표 저장 완료: {RESULT_FILENAME}")

def run_test():
    env = PacmanEnv()
    state_size = 20 * 20 * 5
    action_size = 4

    # 에이전트 객체 생성
    # RANDOM 모델은 state_size가 필요 없지만, 코드 통일성을 위해 인자 처리는 클래스 내부에서 하거나 여기서 분기
    if MODEL_TYPE == "RANDOM":
        agent = AgentClass(action_size)
    else:
        agent = AgentClass(state_size, action_size)

    print(f"\n=== 🎮 {MODEL_TYPE} 모드 테스트 시작 ===")
    print(f"📄 결과 파일: {RESULT_FILENAME}")

    # 모델 가중치 로드 (RANDOM이 아닐 때만)
    if MODEL_TYPE != "RANDOM":
        print(f"📂 모델 파일 로딩 중: {model_filename}")
        try:
            # map_location='cpu'는 GPU가 없는 환경에서도 로드되게 해줍니다.
            agent.model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
            agent.epsilon = 0.0  # 테스트 모드 (탐험 X)
            print(f">>> 로드 성공! AI가 플레이합니다.")
        except FileNotFoundError:
            print(f">>> 🚨 오류: '{model_filename}' 파일이 없습니다.")
            print(f">>> 먼저 train.py에서 해당 모델을 학습시켜 주세요.")
            return
    else:
        print(">>> 🎲 무작위(Random) 플레이어가 준비되었습니다.")

    grid_state = env.reset()
    state = get_one_hot_state(grid_state)
    done = False
    total_reward = 0
    step = 0
    final_wall_hits = 0
    final_coins = 0

    print("--- Game Start ---")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        action = agent.get_action(state)
        next_grid_state, reward, done, info = env.step(action)
        state = get_one_hot_state(next_grid_state)

        final_wall_hits = info['wall_hits']
        final_coins = info['coins_eaten']
        total_reward += reward
        step += 1

        env.render()
        time.sleep(0.03) # 관전 속도

    print("-" * 40)
    print(f"[{MODEL_TYPE}] 테스트 종료")
    print(f"🏆 최종 점수 : {total_reward:.2f}")
    print(f"🦶 생존 스텝 : {step}")
    print(f"💥 벽 충돌수 : {final_wall_hits}")
    print(f"🪙 코인 획득 : {final_coins}")
    print("-" * 40)

    save_result_to_csv(MODEL_TYPE, total_reward, step, final_wall_hits, final_coins)
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    run_test()