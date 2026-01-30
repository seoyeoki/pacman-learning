import numpy as np
import pygame
import torch
import csv
import gc
from datetime import datetime
from pacman_env import PacmanEnv

# =================================================================
# [설정] 주말 풀가동 최적화 세팅
# =================================================================
# 1. 모델: 가장 똑똑했던 DDQN 선택
MODEL_TYPE = "CNN_DDQN"

# 2. 횟수: 주말 동안 충분히 돌도록 50,000으로 상향
EPISODES = 50000
CHECKPOINT_FREQ = 2000  # 2000판마다 저장

# 3. 학습 빈도: 4프레임마다 1번 학습 (속도 3배 향상 + 안정성)
TRAIN_FREQUENCY = 4
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

def main():
    # 파일명 설정
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"../train_result/train_log_{MODEL_TYPE.lower()}_{current_time}.csv"
    model_filename = f"../trained_pth/pacman_{MODEL_TYPE.lower()}.pth"

    print(f"\n{'='*60}")
    print(f"🚀 WEEKEND TRAINING START: {MODEL_TYPE}")
    print(f"🎯 Episodes: {EPISODES}")
    print(f"⚡ Train Frequency: Every {TRAIN_FREQUENCY} steps")
    print(f"📄 Log File: {log_filename}")
    print(f"💾 Model Save: {model_filename}")
    print(f"{'='*60}\n")

    # 환경 및 에이전트 생성
    env = PacmanEnv()
    AgentClass = get_agent_class(MODEL_TYPE)
    agent = AgentClass(action_size=4)

    # [중요] 장기 학습을 위해 엡실론 감쇠율(decay) 미세 조정 (선택 사항)
    # 에피소드가 늘어난 만큼 천천히 줄어들게 설정 (기본값보다 조금 느리게)
    # agent.epsilon_decay = 0.99995  # 필요하다면 주석 해제하여 사용

    # 로그 파일 헤더 작성
    with open(log_filename, 'w', newline='') as f:
        csv.writer(f).writerow(['Episode', 'Score', 'Steps', 'Epsilon', 'Avg_Loss', 'Wall_Hits', 'Coins'])

    try:
        for e in range(EPISODES):
            env.reset()
            state = get_one_hot_state(env.grid, env.pacman_pos, env.ghosts)

            done = False
            total_reward = 0
            step_count = 0
            loss_list = []

            while not done:
                # 윈도우 응답 없음 방지
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return

                action = agent.get_action(state)
                next_grid, reward, done, info = env.step(action)
                next_state = get_one_hot_state(next_grid, env.pacman_pos, env.ghosts)

                agent.remember(state, action, reward, next_state, done)

                # [최적화] 데이터가 2000개 이상 쌓였을 때, 4번 중 1번만 학습
                if len(agent.memory.buffer) > 2000:
                    if step_count % TRAIN_FREQUENCY == 0:
                        loss = agent.train_step()
                        if loss: loss_list.append(loss)

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
        # 학습 완료(혹은 중단) 시 모델 저장 및 리소스 정리
        env.close()
        torch.save(agent.model.state_dict(), model_filename)
        print(f"✨ Finished & Saved: {MODEL_TYPE}")

        # 메모리 정리
        del agent
        del env
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()