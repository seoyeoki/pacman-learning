import numpy as np
import pygame
import torch
import csv # [추가] CSV 저장을 위해 필요
import os

from pacman_env import PacmanEnv
from dqn_agent import DQNAgent

# One-Hot Encoding 함수 (기존 유지)
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
    agent = DQNAgent(state_size, action_size)

    EPISODES = 5000 # [설정] 5000판

    # [추가] 로그 파일 생성
    log_filename = 'training_log.csv'
    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # 헤더(제목) 작성
        writer.writerow(['Episode', 'Score', 'Steps', 'Epsilon', 'Avg_Loss'])

    print("--- DQN Training Start (Logging to CSV) ---")

    for e in range(EPISODES):
        grid_state = env.reset()
        state = get_one_hot_state(grid_state)
        done = False
        total_reward = 0
        step_count = 0

        # [추가] 이번 에피소드의 Loss들을 모을 리스트
        loss_list = []

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            action = agent.get_action(state)
            next_grid_state, reward, done, _ = env.step(action)
            next_state = get_one_hot_state(next_grid_state)

            agent.remember(state, action, reward, next_state, done)

            # [수정] 학습하고 Loss 값 받아오기
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

        # [추가] 평균 Loss 계산 (학습 안 했으면 0)
        avg_loss = np.mean(loss_list) if len(loss_list) > 0 else 0

        # [추가] 파일에 한 줄 기록 (append 모드 'a')
        with open(log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([e+1, total_reward, step_count, agent.epsilon, avg_loss])

        print(f"Episode {e+1}/{EPISODES} | Score: {total_reward:.2f} | Steps: {step_count} | Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.2f}")

    env.close()
    torch.save(agent.model.state_dict(), "pacman_dqn.pth")
    print("Training Finished.")

if __name__ == "__main__":
    main()