import numpy as np
import pygame  # 이벤트 처리를 위해 추가
import torch

from pacman_env import PacmanEnv
from dqn_agent import DQNAgent

def flatten_state(grid):
    """20x20 그리드를 1차원(400,) 배열로 변환"""
    return grid.flatten()

def main():
    env = PacmanEnv()

    # 입력 크기: 20x20 = 400, 행동 크기: 4 (상하좌우)
    state_size = 20 * 20
    action_size = 4

    agent = DQNAgent(state_size, action_size)

    EPISODES = 500  # 총 학습 횟수

    print("--- DQN Training Start ---")

    for e in range(EPISODES):
        # 환경 초기화
        grid_state = env.reset()
        state = flatten_state(grid_state)

        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # -----------------------------------------------------------
            # [수정된 부분] 윈도우 응답 없음 방지를 위한 이벤트 처리 루프
            # -----------------------------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return # 프로그램 즉시 종료
            # -----------------------------------------------------------

            # 1. 행동 선택
            action = agent.get_action(state)

            # 2. 환경 진행
            next_grid_state, reward, done, _ = env.step(action)
            next_state = flatten_state(next_grid_state)

            # 3. 기억 저장
            agent.remember(state, action, reward, next_state, done)

            # 4. 학습 (경험 리플레이)
            agent.train_step()

            state = next_state
            total_reward += reward
            step_count += 1

            # 화면 그리기 조건
            # 'e % 10 == 0'이면 10판마다 한 번만 보여줍니다. (학습 속도 향상)
            # 매번 보고 싶으면 아래 조건을 'if True:'로 바꾸세요.
            if e % 10 == 0:
                env.render()

        # 에피소드 종료 후 엡실론(탐험률) 감소
        agent.update_epsilon()

        print(f"Episode {e+1}/{EPISODES} | Score: {total_reward} | Steps: {step_count} | Epsilon: {agent.epsilon:.2f}")

    # 학습 완료 후 리소스 정리
    env.close()
    print("Training Finished.")

    # 모델 저장 (선택 사항)
    torch.save(agent.model.state_dict(), "pacman_dqn.pth")

if __name__ == "__main__":
    main()