import numpy as np
import pygame
import torch
import csv
import gc
from datetime import datetime
from pacman_env import PacmanEnv
from collections import deque

# =================================================================
# [설정] 주말 풀가동 최적화 세팅
# =================================================================
MODEL_TYPE = "CNN_DDQN"
EPISODES = 50000
CHECKPOINT_FREQ = 2000
TRAIN_FREQUENCY = 4
STACK_SIZE = 4  # 프레임 스택 개수
# =================================================================

# 에이전트 클래스 미리 가져오기
from cnn_model_agent.cnn_dqn_agent import CNNDQNAgent
from cnn_model_agent.cnn_ddqn_agent import CNNDDQNAgent
from cnn_model_agent.cnn_dueling_agent import CNNDuelingAgent

def get_agent_class(model_type):
    if model_type == "CNN_DQN": return CNNDQNAgent
    elif model_type == "CNN_DDQN": return CNNDDQNAgent
    elif model_type == "CNN_DUELING": return CNNDuelingAgent
    else: raise ValueError(f"Unknown Type: {model_type}")

def get_one_hot_state(grid, pacman_pos, ghosts):
    state = np.zeros((5, 20, 20), dtype=np.float32)
    state[0] = (grid == 0) # 길
    state[1] = (grid == 1) # 벽
    state[4] = (grid == 4) # 코인
    state[2][pacman_pos[0], pacman_pos[1]] = 1.0 # 팩맨
    for gr, gc in ghosts:
        state[3][gr, gc] = 1.0 # 유령
    return state

def get_stacked_state(history_buffer, new_state):
    """
    history_buffer: deque 객체 (최근 N개의 상태 저장)
    new_state: 방금 얻은 상태 (5, 20, 20)
    """
    # 1. 버퍼에 새 상태 추가
    history_buffer.append(new_state)

    # 2. 만약 버퍼가 덜 찼으면(초기 상태), 첫 상태로 채움
    while len(history_buffer) < STACK_SIZE:
        history_buffer.append(new_state)

    # 3. 채널 방향(axis=0)으로 합치기
    # 결과 모양: (20, 20, 20) -> (5채널 * 4장)
    return np.concatenate(history_buffer, axis=0)

def main():
    # 파일명 설정
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"../train_result/train_log_{MODEL_TYPE.lower()}_{current_time}.csv"
    model_filename = f"../trained_pth/pacman_{MODEL_TYPE.lower()}.pth"

    print(f"\n{'='*60}")
    print(f"🚀 WEEKEND TRAINING START: {MODEL_TYPE}")
    print(f"🎯 Episodes: {EPISODES}")
    print(f"⚡ Train Frequency: Every {TRAIN_FREQUENCY} steps")
    print(f"📚 Frame Stacking: {STACK_SIZE} frames")
    print(f"📄 Log File: {log_filename}")
    print(f"💾 Model Save: {model_filename}")
    print(f"{'='*60}\n")

    # 환경 및 에이전트 생성
    env = PacmanEnv()
    AgentClass = get_agent_class(MODEL_TYPE)
    agent = AgentClass(action_size=4)

    # 로그 파일 헤더 작성
    with open(log_filename, 'w', newline='') as f:
        csv.writer(f).writerow(['Episode', 'Score', 'Steps', 'Epsilon', 'Avg_Loss', 'Wall_Hits', 'Coins'])

    try:
        # 큐 생성 (maxlen=4로 자동 관리)
        state_buffer = deque(maxlen=STACK_SIZE)

        for e in range(EPISODES):
            env.reset()
            # 초기화 시 버퍼 비우기
            state_buffer.clear()

            # [수정됨] 루프 시작 전 변수 초기화 (매우 중요!)
            done = False
            total_reward = 0
            step_count = 0
            loss_list = []

            # 첫 상태 가져오기
            initial_state = get_one_hot_state(env.grid, env.pacman_pos, env.ghosts)

            # 스택된 상태 만들기 (이게 진짜 state가 됨)
            state = get_stacked_state(state_buffer, initial_state)

            while not done:
                # 윈도우 응답 없음 방지
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return

                action = agent.get_action(state)
                next_grid, reward, done, info = env.step(action)

                # 다음 상태 전처리
                raw_next_state = get_one_hot_state(next_grid, env.pacman_pos, env.ghosts)

                # 스택된 다음 상태 생성
                next_state = get_stacked_state(state_buffer, raw_next_state)

                # [수정됨] 중복 제거: 한 번만 저장해야 함
                agent.remember(state, action, reward, next_state, done)

                # [최적화] 데이터가 2000개 이상 쌓였을 때, 4번 중 1번만 학습
                if len(agent.memory.buffer) > 2000:
                    if step_count % TRAIN_FREQUENCY == 0:
                        loss = agent.train_step()
                        if loss: loss_list.append(loss)

                # 상태 업데이트
                state = next_state
                total_reward += reward
                step_count += 1

            # 에피소드 종료 처리
            agent.update_target_network()
            agent.update_epsilon()

            # 중간 저장
            if (e + 1) % CHECKPOINT_FREQ == 0:
                ckpt_name = f"../trained_pth/pacman_{MODEL_TYPE.lower()}_ep{e+1}.pth"
                torch.save(agent.model.state_dict(), ckpt_name)
                print(f"  💾 [{MODEL_TYPE}] Ep {e+1}: Saved.")

            # 로그 기록
            avg_loss = np.mean(loss_list) if loss_list else 0
            with open(log_filename, 'a', newline='') as f:
                csv.writer(f).writerow([e+1, total_reward, step_count, agent.epsilon, avg_loss, info['wall_hits'], info['coins_eaten']])

            if (e+1) % 100 == 0:
                print(f"[{MODEL_TYPE}] Ep {e+1}/{EPISODES} | Score: {total_reward:.1f} | Eps: {agent.epsilon:.2f} | Loss: {avg_loss:.2f}")

    except KeyboardInterrupt:
        print(f"\n🛑 {MODEL_TYPE} 학습 강제 중단됨!")

    finally:
        env.close()
        torch.save(agent.model.state_dict(), model_filename)
        print(f"✨ Finished & Saved: {MODEL_TYPE}")

        del agent
        del env
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()