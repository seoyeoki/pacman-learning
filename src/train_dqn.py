import numpy as np
import pygame
import torch
import csv
import os

from pacman_env import PacmanEnv
from dqn_agent import DQNAgent

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

    EPISODES = 5000

    log_filename = 'training_log.csv'
    # 파일 생성 및 헤더 작성
    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # 헤더 7개 확인!
        writer.writerow(['Episode', 'Score', 'Steps', 'Epsilon', 'Avg_Loss', 'Wall_Hits', 'Coins'])

    print("--- DQN Training Start ---")

    for e in range(EPISODES):
        grid_state = env.reset()
        state = get_one_hot_state(grid_state)
        done = False
        total_reward = 0
        step_count = 0

        # [수정 1] 이 리스트 초기화가 반드시 있어야 합니다!
        loss_list = []

        final_wall_hits = 0
        final_coins = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            action = agent.get_action(state)

            # 환경 진행 (info 받아오기)
            next_grid_state, reward, done, info = env.step(action)
            next_state = get_one_hot_state(next_grid_state)

            # 통계 갱신
            final_wall_hits = info['wall_hits']
            final_coins = info['coins_eaten']

            agent.remember(state, action, reward, next_state, done)

            # 학습 및 오차 기록
            loss = agent.train_step()
            if loss is not None:
                loss_list.append(loss) # 이제 에러 안 남

            state = next_state
            total_reward += reward
            step_count += 1

            if e % 100 == 0:
                env.render()

        agent.update_target_network()
        agent.update_epsilon()

        # 평균 Loss 계산
        avg_loss = np.mean(loss_list) if len(loss_list) > 0 else 0

        # [수정 2] CSV 기록할 때 벽/코인 횟수도 같이 넣어줘야 합니다!
        with open(log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            # 순서: Episode, Score, Steps, Epsilon, Loss, Wall, Coins
            writer.writerow([e+1, total_reward, step_count, agent.epsilon, avg_loss, final_wall_hits, final_coins])

        print(f"Ep {e+1}/{EPISODES} | Score: {total_reward:.2f} | Wall: {final_wall_hits} | Coins: {final_coins} | Eps: {agent.epsilon:.2f}")

    env.close()
    torch.save(agent.model.state_dict(), "pacman_dqn.pth")
    print("Training Finished.")

if __name__ == "__main__":
    main()