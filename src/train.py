import numpy as np
import pygame
import torch
import csv
from datetime import datetime # [추가] 날짜 기능을 위해 필요
from pacman_env import PacmanEnv

# =================================================================
# [설정] 모델 타입 선택
MODEL_TYPE = "DDQN"
# =================================================================

# 1. 현재 시간 구하기 (예: 20240521_153000)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 2. 파일명에 시간 포함시키기
# 로그 파일: 매번 새로운 파일 생성 (기록 보존용)
log_filename = f"train_log_{MODEL_TYPE}_{current_time}.csv"

# 모델 파일: 편의상 최신 파일 하나로 덮어쓰기 유지 (test.py가 찾기 쉽게)
# (원하시면 모델 파일명에도 시간을 붙일 수 있지만, 그러면 테스트할 때마다 파일명을 수정해야 합니다.)
model_filename = f"pacman_{MODEL_TYPE.lower()}.pth"

# 모델 선택 로직 (기존과 동일)
if MODEL_TYPE == "DQN":
    from src.model_agent.dqn_agent import DQNAgent as Agent
elif MODEL_TYPE == "DDQN":
    from src.model_agent.ddqn_agent import DDQNAgent as Agent
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

def main():
    env = PacmanEnv()
    state_size = 20 * 20 * 5
    action_size = 4
    agent = Agent(state_size, action_size)
    EPISODES = 5000

    print(f"--- Training Start: {MODEL_TYPE} ---")
    print(f"📄 로그 파일: {log_filename}") # 바뀐 파일명 확인
    print(f"💾 모델 저장: {model_filename}")

    # CSV 파일 생성
    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Score', 'Steps', 'Epsilon', 'Avg_Loss', 'Wall_Hits', 'Coins'])

    for e in range(EPISODES):
        grid_state = env.reset()
        state = get_one_hot_state(grid_state)
        done = False
        total_reward = 0
        step_count = 0
        loss_list = []
        final_wall_hits = 0
        final_coins = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            action = agent.get_action(state)
            next_grid_state, reward, done, info = env.step(action)
            next_state = get_one_hot_state(next_grid_state)

            final_wall_hits = info['wall_hits']
            final_coins = info['coins_eaten']

            agent.remember(state, action, reward, next_state, done)

            # 4스텝마다 학습 (속도 최적화)
            if len(agent.memory) > 64 and step_count % 4 == 0:
                loss = agent.train_step()
                if loss is not None:
                    loss_list.append(loss)

            state = next_state
            total_reward += reward
            step_count += 1

            # 화면은 100판마다 (속도 최적화)
            if (e + 1) % 100 == 0:
                env.render()

        agent.update_target_network()
        agent.update_epsilon()

        avg_loss = np.mean(loss_list) if len(loss_list) > 0 else 0

        # 로그 저장
        with open(log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([e+1, total_reward, step_count, agent.epsilon, avg_loss, final_wall_hits, final_coins])

        if (e + 1) % 100 == 0:
            print(f"[{MODEL_TYPE}] Ep {e+1}/{EPISODES} | Score: {total_reward:.2f} | Wall: {final_wall_hits} | Coins: {final_coins} | Eps: {agent.epsilon:.2f}")

    env.close()
    torch.save(agent.model.state_dict(), model_filename)
    print(f"\nTraining Finished! Saved to {log_filename}")

if __name__ == "__main__":
    main()