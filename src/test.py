import pygame
import torch
import time
import numpy as np
import csv
from datetime import datetime
from pacman_env import PacmanEnv

# =================================================================
# [설정] 테스트할 모델 타입
# "CNN_DQN", "CNN_DDQN", "CNN_DUELING"
MODEL_TYPE = "RANDOM"
NUM_TEST_EPISODES = 10
# =================================================================

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_FILENAME = f"../test_result/test_summary_{MODEL_TYPE}_{current_time}.csv"

# 모델 로드 로직
if MODEL_TYPE == "CNN_DQN":
    from cnn_model_agent.cnn_dqn_agent import CNNDQNAgent as AgentClass
    model_filename = "../trained_pth/pacman_cnn_dqn.pth"
elif MODEL_TYPE == "CNN_DDQN":
    from cnn_model_agent.cnn_ddqn_agent import CNNDDQNAgent as AgentClass
    model_filename = "../trained_pth/pacman_cnn_ddqn.pth"
elif MODEL_TYPE == "CNN_DUELING":import pygame
import torch
import time
import numpy as np
import csv
from datetime import datetime
from pacman_env import PacmanEnv

# =================================================================
# [설정] 테스트할 모델 타입
# 옵션: "CNN_DQN", "CNN_DDQN", "CNN_DUELING", "RANDOM"
MODEL_TYPE = "RANDOM"
NUM_TEST_EPISODES = 10
# =================================================================

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_FILENAME = f"../test_result/test_summary_{MODEL_TYPE}_{current_time}.csv"

# 모델 로드 로직
model_filename = None # 초기화

if MODEL_TYPE == "CNN_DQN":
    from cnn_model_agent.cnn_dqn_agent import CNNDQNAgent as AgentClass
    model_filename = "../trained_pth/pacman_cnn_dqn.pth"
elif MODEL_TYPE == "CNN_DDQN":
    from cnn_model_agent.cnn_ddqn_agent import CNNDDQNAgent as AgentClass
    model_filename = "../trained_pth/pacman_cnn_ddqn.pth"
elif MODEL_TYPE == "CNN_DUELING":
    from cnn_model_agent.cnn_dueling_agent import CNNDuelingAgent as AgentClass
    model_filename = "../trained_pth/pacman_cnn_dueling.pth"
elif MODEL_TYPE == "RANDOM":
    # 위에서 만든 random_agent.py가 있어야 합니다.
    from cnn_model_agent.random_agent import RandomAgent as AgentClass
    model_filename = None # 랜덤은 불러올 파일 없음
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

# CNN용 상태 전처리 (Train과 동일)
def get_one_hot_state(grid, pacman_pos, ghosts):
    state = np.zeros((5, 20, 20), dtype=np.float32)
    state[0] = (grid == 0)
    state[1] = (grid == 1)
    state[4] = (grid == 4)
    pr, pc = pacman_pos
    state[2][pr, pc] = 1.0
    for gr, gc in ghosts:
        state[3][gr, gc] = 1.0
    return state

def run_episode(env, agent, episode_idx):
    env.reset()
    state = get_one_hot_state(env.grid, env.pacman_pos, env.ghosts)
    done = False
    total_reward = 0
    step = 0

    pygame.display.set_caption(f"{MODEL_TYPE} Test - Ep {episode_idx}")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        action = agent.get_action(state)
        next_grid, reward, done, info = env.step(action)
        state = get_one_hot_state(next_grid, env.pacman_pos, env.ghosts)

        total_reward += reward
        step += 1

        # 렌더링 (너무 빠르면 time.sleep 주석 해제)
        env.render()
        # time.sleep(0.01)

    return {
        'episode': episode_idx,
        'score': total_reward,
        'steps': step,
        'wall_hits': info['wall_hits'],
        'coins': info['coins_eaten']
    }

def main():
    env = PacmanEnv()
    action_size = 4
    agent = AgentClass(action_size)

    # [수정됨] 모델 로드 로직 (RANDOM일 때는 건너뜀)
    if MODEL_TYPE != "RANDOM":
        print(f"Loading Model: {model_filename}")
        try:
            # weights_only=True 추가 (경고 방지)
            agent.model.load_state_dict(torch.load(model_filename, map_location='cpu', weights_only=True))
            agent.epsilon = 0.0 # 탐험 끄기 (순수 실력 테스트)
            print("✅ Model Loaded Successfully!")
        except FileNotFoundError:
            print(f"❌ Error: Model file not found at {model_filename}")
            print("Please train the model first.")
            return
    else:
        print("🎲 Random Agent Selected (No model to load)")

    history = []
    print(f"\n--- Start Testing ({NUM_TEST_EPISODES} Episodes) ---")

    for i in range(1, NUM_TEST_EPISODES + 1):
        res = run_episode(env, agent, i)
        history.append(res)
        print(f"Ep {i} | Score: {res['score']:.1f} | Wall: {res['wall_hits']} | Coins: {res['coins']}")

    # --- 최종 결과 요약 출력 ---
    scores = [h['score'] for h in history]
    walls = [h['wall_hits'] for h in history]
    coins = [h['coins'] for h in history]
    steps = [h['steps'] for h in history]

    print("\n" + "="*50)
    print(f"   📊 [ {MODEL_TYPE} ] 최종 성적표 (총 {NUM_TEST_EPISODES}회)")
    print("="*50)
    print(f"   🏆 평균 점수 (Score) : {np.mean(scores):.2f}  (Max: {np.max(scores):.2f})")
    print(f"   🪙 평균 코인 (Coins) : {np.mean(coins):.1f}   (Max: {np.max(coins)})")
    print(f"   💥 평균 충돌 (Walls) : {np.mean(walls):.1f}   (Min: {np.min(walls)})")
    print(f"   🦶 평균 스텝 (Steps) : {np.mean(steps):.1f}")
    print("-" * 50)

    # CSV 저장 (선택사항)
    with open(RESULT_FILENAME, 'w', newline='') as f:
         writer = csv.writer(f)
         writer.writerow(['Episode', 'Score', 'Wall_Hits', 'Coins', 'Steps'])
         for h in history:
             writer.writerow([h['episode'], h['score'], h['wall_hits'], h['coins'], h['steps']])
    print(f"📁 Log saved to {RESULT_FILENAME}")

if __name__ == "__main__":
    main()
    from cnn_model_agent.cnn_dueling_agent import CNNDuelingAgent as AgentClass
    model_filename = "../trained_pth/pacman_cnn_dueling.pth"
elif MODEL_TYPE == "RANDOM":
    from cnn_model_agent.random_agent import RandomAgent as AgentClass
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

# CNN용 상태 전처리 (Train과 동일)
def get_one_hot_state(grid, pacman_pos, ghosts):
    state = np.zeros((5, 20, 20), dtype=np.float32)
    state[0] = (grid == 0)
    state[1] = (grid == 1)
    state[4] = (grid == 4)
    pr, pc = pacman_pos
    state[2][pr, pc] = 1.0
    for gr, gc in ghosts:
        state[3][gr, gc] = 1.0
    return state

def run_episode(env, agent, episode_idx):
    env.reset()
    state = get_one_hot_state(env.grid, env.pacman_pos, env.ghosts)
    done = False
    total_reward = 0
    step = 0

    pygame.display.set_caption(f"{MODEL_TYPE} Test - Ep {episode_idx}")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        action = agent.get_action(state)
        next_grid, reward, done, info = env.step(action)
        state = get_one_hot_state(next_grid, env.pacman_pos, env.ghosts)

        total_reward += reward
        step += 1
        env.render()
        # time.sleep(0.01) # 너무 빠르면 주석 해제

    return {
        'episode': episode_idx,
        'score': total_reward,
        'steps': step,
        'wall_hits': info['wall_hits'],
        'coins': info['coins_eaten']
    }

def main():
    env = PacmanEnv()
    action_size = 4
    agent = AgentClass(action_size)

    print(f"Loading Model: {model_filename}")
    try:
        agent.model.load_state_dict(torch.load(model_filename, map_location='cpu'))
        agent.epsilon = 0.0 # 탐험 끄기
        print("Model Loaded Successfully!")
    except FileNotFoundError:
        print("Model file not found. Please train first.")
        return

    history = []
    for i in range(1, NUM_TEST_EPISODES + 1):
        res = run_episode(env, agent, i)
        history.append(res)
        print(f"Ep {i} | Score: {res['score']:.1f} | Wall: {res['wall_hits']}")

    # 결과 집계 출력 (생략, 기존과 동일)
    # ...

if __name__ == "__main__":
    main()