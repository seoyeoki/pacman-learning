import numpy as np
import pygame
import torch
import csv
import os
from pacman_env import PacmanEnv

# =================================================================
# [설정] 여기에 원하는 모델 이름을 적으세요.
# 옵션: "DQN", "DDQN", "DUELING"
MODEL_TYPE = "DDQN"
# =================================================================

# 1. 파일 이름 자동 생성 (소문자로 변환)
# 예: DDQN -> "log_ddqn.csv", "pacman_ddqn.pth"
log_filename = f"log_{MODEL_TYPE.lower()}.csv"
model_filename = f"pacman_{MODEL_TYPE.lower()}.pth"

# 2. 모델 타입에 맞는 에이전트 불러오기
if MODEL_TYPE == "DQN":
    from dqn_agent import DQNAgent as Agent
    print(f">>> ⚡ [Standard DQN] 모드로 설정됨.")

elif MODEL_TYPE == "DDQN":
    from ddqn_agent import DDQNAgent as Agent
    print(f">>> 🔥 [Double DQN] 모드로 설정됨.")

elif MODEL_TYPE == "DUELING":
    from dueling_agent import DuelingAgent as Agent
    print(f">>> ⚔️ [Dueling DQN] 모드로 설정됨.")

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

def main():
    env = PacmanEnv()
    state_size = 20 * 20 * 5
    action_size = 4

    agent = Agent(state_size, action_size)

    EPISODES = 5000

    print(f"--- Training Start: {MODEL_TYPE} ---")
    print(f"📄 로그 저장: {log_filename}")
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
            loss = agent.train_step()
            if loss is not None:
                loss_list.append(loss)

            state = next_state
            total_reward += reward
            step_count += 1

            if e % 100 == 0:
                env.render()

        agent.update_target_network()
        agent.update_epsilon()

        avg_loss = np.mean(loss_list) if len(loss_list) > 0 else 0

        # 로그 파일에 기록 (위에서 만든 log_filename 사용)
        with open(log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([e+1, total_reward, step_count, agent.epsilon, avg_loss, final_wall_hits, final_coins])

        print(f"[{MODEL_TYPE}] Ep {e+1}/{EPISODES} | Score: {total_reward:.2f} | Wall: {final_wall_hits} | Coins: {final_coins} | Eps: {agent.epsilon:.2f}")

    env.close()

    # 모델 파일 저장 (위에서 만든 model_filename 사용)
    torch.save(agent.model.state_dict(), model_filename)
    print(f"Training Finished. Model saved as {model_filename}")

if __name__ == "__main__":
    main()