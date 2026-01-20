import numpy as np
import pygame
import torch
from pacman_env import PacmanEnv
from dqn_agent import DQNAgent

def get_one_hot_state(grid):
    """
    20x20 그리드 -> One-Hot Encoding -> 1차원 배열(2000개)
    0:빈칸, 1:벽, 2:팩맨, 3:유령, 4:코인
    """
    # 채널 5개, 20x20 크기
    state_one_hot = np.zeros((5, 20, 20), dtype=np.float32)

    state_one_hot[0] = (grid == 0) # 빈칸
    state_one_hot[1] = (grid == 1) # 벽
    state_one_hot[2] = (grid == 2) # 팩맨
    state_one_hot[3] = (grid == 3) # 유령
    state_one_hot[4] = (grid == 4) # 코인

    return state_one_hot.flatten() # 20*20*5 = 2000

def main():
    env = PacmanEnv()

    # 입력 크기 변경: 20x20 -> 20x20x5 = 2000
    state_size = 20 * 20 * 5
    action_size = 4

    agent = DQNAgent(state_size, action_size)

    EPISODES = 500  # 테스트용 500판 (원하면 3000~5000 추천)

    print("--- DQN Training Start (One-Hot + Target Network) ---")

    for e in range(EPISODES):
        grid_state = env.reset()
        state = get_one_hot_state(grid_state) # [변경] 원-핫 인코딩

        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # 윈도우 응답 없음 방지
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            # 행동 선택
            action = agent.get_action(state)

            # 환경 진행
            next_grid_state, reward, done, _ = env.step(action)
            next_state = get_one_hot_state(next_grid_state) # [변경] 원-핫 인코딩

            # 기억 저장 및 학습
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward
            step_count += 1

            # 100판마다 화면 렌더링
            if e % 100 == 0:
                env.render()

        # 에피소드 종료 후: 타겟 네트워크 업데이트 & 엡실론 감소
        agent.update_target_network()
        agent.update_epsilon()

        print(f"Episode {e+1}/{EPISODES} | Score: {total_reward:.2f} | Steps: {step_count} | Epsilon: {agent.epsilon:.2f}")

    env.close()
    torch.save(agent.model.state_dict(), "pacman_dqn.pth")
    print("Training Finished.")

if __name__ == "__main__":
    main()