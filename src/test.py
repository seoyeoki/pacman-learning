import pygame
import torch
import time
import numpy as np
import csv
from datetime import datetime
from pacman_env import PacmanEnv

# =================================================================
# [설정] 테스트할 모델 타입을 선택하세요.
# =================================================================
MODEL_TYPE = "DUELING"  # "DQN", "DDQN", "DUELING", "RANDOM"
NUM_TEST_EPISODES = 10  # 테스트 반복 횟수
RENDER_DELAY = 0.01     # 관전 속도 (빠른 진행을 위해 0.01 추천)
# =================================================================

# 1. 파일명 생성 (타임스탬프)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_FILENAME = f"../test_result/test_summary_{MODEL_TYPE}_{current_time}.csv"

# --- 모델 타입에 따른 클래스 및 파일 설정 ---
if MODEL_TYPE == "RANDOM":
    from model_agent.random_agent import RandomAgent as AgentClass
    model_filename = None
elif MODEL_TYPE == "DQN":
    from model_agent.dqn_agent import DQNAgent as AgentClass
    model_filename = "../trained_pth/pacman_dqn.pth"
elif MODEL_TYPE == "DDQN":
    from model_agent.ddqn_agent import DDQNAgent as AgentClass
    model_filename = "../trained_pth/pacman_ddqn.pth"
elif MODEL_TYPE == "DUELING":
    from model_agent.dueling_agent import DuelingAgent as AgentClass
    model_filename = "../trained_pth/pacman_dueling.pth"
else:
    raise ValueError(f"지원하지 않는 모델 타입입니다: {MODEL_TYPE}")

def get_one_hot_state(grid):
    state_one_hot = np.zeros((5, 20, 20), dtype=np.float32)
    state_one_hot[0] = (grid == 0) # 길
    state_one_hot[1] = (grid == 1) # 벽
    state_one_hot[2] = (grid == 2) # 팩맨
    state_one_hot[3] = (grid == 3) # 유령
    state_one_hot[4] = (grid == 4) # 코인
    return state_one_hot.flatten()

def save_summary_to_csv(results):
    """10회 테스트 결과를 CSV로 저장"""
    with open(RESULT_FILENAME, 'w', newline='') as f:
        writer = csv.writer(f)
        # 헤더 작성
        writer.writerow(['Episode', 'Timestamp', 'Model_Type', 'Score', 'Steps', 'Wall_Hits', 'Coins'])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for res in results:
            writer.writerow([
                res['episode'],
                timestamp,
                MODEL_TYPE,
                res['score'],
                res['steps'],
                res['wall_hits'],
                res['coins']
            ])

    print(f"💾 [저장 완료] 상세 기록이 저장되었습니다: {RESULT_FILENAME}")

def run_episode(env, agent, episode_idx):
    """한 번의 에피소드를 실행하고 결과를 반환하는 함수"""
    grid_state = env.reset()
    state = get_one_hot_state(grid_state)
    done = False
    total_reward = 0
    step = 0
    final_wall_hits = 0
    final_coins = 0

    # 윈도우 제목에 현재 진행상황 표시
    pygame.display.set_caption(f"{MODEL_TYPE} Test - Episode {episode_idx}/{NUM_TEST_EPISODES}")

    while not done:
        # 이벤트 처리 (강제 종료 방지)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        action = agent.get_action(state)
        next_grid_state, reward, done, info = env.step(action)
        state = get_one_hot_state(next_grid_state)

        final_wall_hits = info['wall_hits']
        final_coins = info['coins_eaten']
        total_reward += reward
        step += 1

        env.render()
        if RENDER_DELAY > 0:
            time.sleep(RENDER_DELAY)

    return {
        'episode': episode_idx,
        'score': total_reward,
        'steps': step,
        'wall_hits': final_wall_hits,
        'coins': final_coins
    }

def run_test_batch():
    env = PacmanEnv()
    state_size = 20 * 20 * 5
    action_size = 4

    # 1. 에이전트 생성
    if MODEL_TYPE == "RANDOM":
        agent = AgentClass(action_size)
    else:
        agent = AgentClass(state_size, action_size)

    print(f"\n=== 🎮 {MODEL_TYPE} 모델 10회 연속 테스트 시작 ===")

    # 2. 모델 로드 (1회만 수행)
    if MODEL_TYPE != "RANDOM":
        print(f"📂 모델 불러오는 중: {model_filename}")
        try:
            agent.model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
            agent.epsilon = 0.0  # 탐험 끄기 (Greedy Action)
            print(f">>> 로드 성공! 테스트를 시작합니다.")
        except FileNotFoundError:
            print(f">>> 🚨 오류: '{model_filename}' 파일이 없습니다. 학습을 먼저 진행하세요.")
            return
    else:
        print(">>> 🎲 Random Agent 준비 완료.")

    # 3. 10회 반복 실행
    history = []

    for i in range(1, NUM_TEST_EPISODES + 1):
        print(f"\n▶ Episode {i}/{NUM_TEST_EPISODES} 진행 중...", end="\r")
        result = run_episode(env, agent, i)
        history.append(result)

        # 짧은 요약 출력
        print(f"▶ Episode {i:02d} | Score: {result['score']:.1f} | Coins: {result['coins']} | Walls: {result['wall_hits']}")
        time.sleep(0.5) # 에피소드 간 짧은 대기

    env.close()

    # 4. 결과 집계 및 출력
    scores = [r['score'] for r in history]
    steps = [r['steps'] for r in history]
    walls = [r['wall_hits'] for r in history]
    coins = [r['coins'] for r in history]

    print("\n" + "="*50)
    print(f"   📊 [ {MODEL_TYPE} ] 최종 성적표 (총 {NUM_TEST_EPISODES}회)")
    print("="*50)
    print(f"   🏆 평균 점수 (Score) : {np.mean(scores):.2f}  (Max: {np.max(scores):.2f})")
    print(f"   🪙 평균 코인 (Coins) : {np.mean(coins):.1f}   (Max: {np.max(coins)})")
    print(f"   💥 평균 충돌 (Walls) : {np.mean(walls):.1f}   (Min: {np.min(walls)})")
    print(f"   🦶 평균 스텝 (Steps) : {np.mean(steps):.1f}")
    print("-" * 50)

    # 5. CSV 저장
    save_summary_to_csv(history)

if __name__ == "__main__":
    run_test_batch()