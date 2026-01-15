import numpy as np
import pygame
import torch
from pacman_env import PacmanEnv
from dqn_agent import DQNAgent

# [핵심 수정] 숫자를 벡터로 바꾸는 원-핫 인코딩 함수
def get_one_hot_state(grid):
    """
    20x20 그리드 -> 20x20x5 (채널 5개) -> 1차원 배열(2000개)로 변환
    0:빈칸, 1:벽, 2:팩맨, 3:유령, 4:코인
    """
    state_one_hot = np.zeros((5, 20, 20), dtype=np.float32)

    # 각 채널별로 1을 찍어줍니다.
    state_one_hot[0] = (grid == 0) # 빈칸 채널
    state_one_hot[1] = (grid == 1) # 벽 채널
    state_one_hot[2] = (grid == 2) # 팩맨 채널
    state_one_hot[3] = (grid == 3) # 유령 채널
    state_one_hot[4] = (grid == 4) # 코인 채널

    return state_one_hot.flatten() # 20*20*5 = 2000개 입력

def main():
    env = PacmanEnv()

    # [수정] 입력 크기가 400이 아니라 2000이 됩니다.
    # 20(가로) * 20(세로) * 5(종류: 빈칸,벽,팩맨,유령,코인)
    state_size = 20 * 20 * 5
    action_size = 4

    agent = DQNAgent(state_size, action_size)

    EPISODES = 500

    # [설정] 500판 기준에 맞춘 Epsilon Decay 재설정 (조금 더 천천히 줄어들게)
    agent.epsilon_decay = 0.995

    print("--- DQN Training Start (One-Hot + Target Network) ---")

    for e in range(EPISODES):
        grid_state = env.reset()
        state = get_one_hot_state(grid_state) # [수정] 원-핫 인코딩 사용

        done = False
        total_reward = 0
        step_count = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            action = agent.get_action(state)

            next_grid_state, reward, done, _ = env.step(action)
            next_state = get_one_hot_state(next_grid_state) # [수정] 원-핫

            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward
            step_count += 1

            if e % 20 == 0: # 20판마다 렌더링 (속도업)
                env.render()

        # [핵심 수정] 에피소드가 끝날 때마다 타겟 네트워크를 업데이트 (정답지 갱신)
        agent.update_target_network()
        agent.update_epsilon()

        print(f"Episode {e+1}/{EPISODES} | Score: {total_reward} | Steps: {step_count} | Epsilon: {agent.epsilon:.2f}")

    env.close()
    torch.save(agent.model.state_dict(), "pacman_dqn.pth")
    print("Training Finished.")

if __name__ == "__main__":
    main()