import numpy as np
import pygame
import torch
import csv
import os
from pacman_env import PacmanEnv

# =================================================================
# [설정] 모델 타입 선택
# 옵션: "DQN", "DDQN", "DUELING"
MODEL_TYPE = "DDQN"
# =================================================================

# 파일명 자동 생성
log_filename = f"log_{MODEL_TYPE.lower()}.csv"
model_filename = f"pacman_{MODEL_TYPE.lower()}.pth"

# 모델 선택 로직
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

def main():
    # 학습 속도를 높이려면 render를 아예 안 하는 게 좋습니다.
    # 화면을 안 띄우고 싶다면 PacmanEnv() 내부에서 pygame.display.set_mode를 주석 처리하거나
    # render() 함수 호출을 아예 지워야 하지만, 일단 여기서는 호출 빈도만 줄입니다.
    env = PacmanEnv()
    state_size = 20 * 20 * 5
    action_size = 4

    agent = Agent(state_size, action_size)

    EPISODES = 5000

    print(f"--- Training Start: {MODEL_TYPE} ---")
    print(f"📄 로그는 '{log_filename}' 파일에만 저장됩니다.")
    print("🚀 학습 중... (터미널 출력은 100 에피소드마다 갱신됩니다)")

    # CSV 파일 초기화
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

            # [옵션] 학습 화면도 100판에 한 번만, 혹은 아예 주석 처리해서 끄세요.
            if (e + 1) % 100 == 0:
                env.render()

        agent.update_target_network()
        agent.update_epsilon()

        avg_loss = np.mean(loss_list) if len(loss_list) > 0 else 0

        # 1. 로그 파일 저장은 매 판 수행 (데이터 확보용)
        with open(log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([e+1, total_reward, step_count, agent.epsilon, avg_loss, final_wall_hits, final_coins])

        # 2. 터미널 출력은 100판마다 한 번만 (생존 신고용)
        if (e + 1) % 100 == 0:
            print(f"[{MODEL_TYPE}] Ep {e+1}/{EPISODES} | Score: {total_reward:.2f} | Wall: {final_wall_hits} | Coins: {final_coins} | Eps: {agent.epsilon:.2f}")

    env.close()
    torch.save(agent.model.state_dict(), model_filename)
    print(f"\nTraining Finished! Model saved as {model_filename}")

if __name__ == "__main__":
    main()