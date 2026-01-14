import pygame
import torch
import time
import random
from pacman_env import PacmanEnv
from dqn_agent import DQNAgent

# 20x20 그리드 -> 400 입력
def flatten_state(grid):
    return grid.flatten()

def run_test(mode='trained'):
    """
    mode: 'random' (학습 전) 또는 'trained' (학습 후)
    """
    env = PacmanEnv()
    state_size = 20 * 20
    action_size = 4

    agent = DQNAgent(state_size, action_size)

    if mode == 'trained':
        print("\n=== 🧠 학습된 AI(After) 로딩 중... ===")
        try:
            # 저장된 모델 가중치 불러오기
            agent.model.load_state_dict(torch.load("pacman_dqn.pth", map_location=torch.device('cpu')))
            agent.epsilon = 0.0  # 탐험(랜덤 행동)을 끄고, 배운 대로만 행동
            print(">>> 모델 로드 성공! AI가 플레이합니다.")
        except FileNotFoundError:
            print(">>> 🚨 오류: 'pacman_dqn.pth' 파일을 찾을 수 없습니다!")
            print(">>> train_dqn.py를 실행해서 먼저 모델을 만들어주세요.")
            return
    else:
        print("\n=== 🎲 랜덤 팩맨(Before) 시작... ===")
        # 아무것도 로드하지 않음 (초기화된 상태 = 바보)

    # 테스트 게임 시작 (1판만)
    grid_state = env.reset()
    state = flatten_state(grid_state)
    done = False
    total_reward = 0
    step = 0

    while not done:
        # 이벤트 처리 (창 닫힘 방지)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        # 행동 선택
        if mode == 'random':
            action = random.choice([0, 1, 2, 3]) # 완전 랜덤
        else:
            action = agent.get_action(state)   # AI 판단

        # 환경 진행
        next_grid_state, reward, done, _ = env.step(action)
        state = flatten_state(next_grid_state)
        total_reward += reward
        step += 1

        # 화면 그리기 (테스트니까 매 프레임 그리기)
        env.render()

        # 너무 빠르면 눈에 안 보이니 약간 딜레이 (선택 사항)
        # time.sleep(0.05)

    print(f"[{mode.upper()}] 게임 종료! 점수: {total_reward}, 생존 시간: {step} 스텝")
    time.sleep(1) # 결과 확인용 대기
    env.close()

if __name__ == "__main__":
    # 1. 학습 전 (Before) 확인
    run_test(mode='random')

    # 2. 학습 후 (After) 확인
    run_test(mode='trained')