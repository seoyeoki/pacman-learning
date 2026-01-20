import numpy as np
import pygame
import torch
import csv
import os
from pacman_env import PacmanEnv

# =================================================================
# [설정] 사용할 모델을 문자열로 지정하세요.
# 옵션: "DQN", "DDQN", "DUELING"
MODEL_TYPE = "DDQN"
# =================================================================

# 모델 타입에 따라 클래스와 파일명 자동 설정
if MODEL_TYPE == "DQN":
    from dqn_agent import DQNAgent as Agent
    print(f">>> ⚡ [Standard DQN] 모드로 학습을 준비합니다.")

elif MODEL_TYPE == "DDQN":
    from ddqn_agent import DDQNAgent as Agent
    print(f">>> 🔥 [Double DQN] 모드로 학습을 준비합니다.")

elif MODEL_TYPE == "DUELING":
    # dueling_agent.py가 있어야 실행됩니다 (아래 3번 코드 참고)
    from dueling_agent import DuelingAgent as Agent
    print(f">>> ⚔️ [Dueling DQN] 모드로 학습을 준비합니다.")

else:
    raise ValueError(f"지원하지 않는 모델 타입입니다: {MODEL_TYPE}")

# 파일명 자동 생성 (예: pacman_dqn.pth, log_ddqn.csv)
model_filename = f"pacman_{MODEL_TYPE.lower()}.pth"
log_filename = f"log_{MODEL_TYPE.lower()}.csv"


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

    # 선택된 Agent 클래스로 인스턴스 생성
    agent = Agent(state_size, action_size)

    EPISODES = 5000

    print(f"--- Training Start: {MODEL_TYPE} ---")
    print(f"Logs will be saved to: {log_filename}")
    print(f"Model will be saved to: {model_filename}")

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

        with open(log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([e+1, total_reward, step_count, agent.epsilon, avg_loss, final_wall_hits, final_coins])

        print(f"[{MODEL_TYPE}] Ep {e+1}/{EPISODES} | Score: {total_reward:.2f} | Wall: {final_wall_hits} | Coins: {final_coins} | Eps: {agent.epsilon:.2f}")

    env.close()
    torch.save(agent.model.state_dict(), model_filename)
    print(f"Training Finished. Model saved as {model_filename}")

if __name__ == "__main__":
    main()