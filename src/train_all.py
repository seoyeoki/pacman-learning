import numpy as np
import pygame
import torch
import csv
import gc
from datetime import datetime
from pacman_env import PacmanEnv

# =================================================================
# [설정] 순차적으로 학습할 모델 목록
# =================================================================
MODELS_TO_TRAIN = ["CNN_DQN", "CNN_DDQN", "CNN_DUELING"]
EPISODES_PER_MODEL = 5000  # <--- 20000에서 5000으로 수정! (핵심)
CHECKPOINT_FREQ = 1000     # 1000판마다 저장 (예선전이니까 자주)
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

def train_single_model(model_type):
    """하나의 모델을 처음부터 끝까지 학습하는 함수"""

    # 각 실행마다 고유한 타임스탬프 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"../train_result/train_log_{model_type.lower()}_{current_time}.csv"
    model_filename = f"../trained_pth/pacman_{model_type.lower()}.pth"

    print(f"\n{'='*60}")
    print(f"🚀 START TRAINING: {model_type}")
    print(f"📄 Log File: {log_filename}")
    print(f"💾 Model Save: {model_filename}")
    print(f"{'='*60}\n")

    # 환경 및 에이전트 생성
    env = PacmanEnv()
    AgentClass = get_agent_class(model_type)
    agent = AgentClass(action_size=4)

    # 로그 파일 헤더 작성
    with open(log_filename, 'w', newline='') as f:
        csv.writer(f).writerow(['Episode', 'Score', 'Steps', 'Epsilon', 'Avg_Loss', 'Wall_Hits', 'Coins'])

    try:
        for e in range(EPISODES_PER_MODEL):
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
                        return False # 강제 종료 시그널

                action = agent.get_action(state)
                next_grid, reward, done, info = env.step(action)
                next_state = get_one_hot_state(next_grid, env.pacman_pos, env.ghosts)

                agent.remember(state, action, reward, next_state, done)

                if len(agent.memory.buffer) > 1000:
                    loss = agent.train_step()
                    if loss: loss_list.append(loss)

                state = next_state
                total_reward += reward
                step_count += 1

            # 에피소드 종료 처리
            agent.update_target_network()
            agent.update_epsilon()

            # [수정됨] 5000판마다 중간 저장
            if (e + 1) % CHECKPOINT_FREQ == 0:
                ckpt_name = f"../trained_pth/pacman_{model_type.lower()}_ep{e+1}.pth"
                torch.save(agent.model.state_dict(), ckpt_name)
                print(f"  💾 [{model_type}] Ep {e+1}: Checkpoint saved.")

            # 로그 기록
            avg_loss = np.mean(loss_list) if loss_list else 0
            with open(log_filename, 'a', newline='') as f:
                csv.writer(f).writerow([e+1, total_reward, step_count, agent.epsilon, avg_loss, info['wall_hits'], info['coins_eaten']])

            if (e+1) % 100 == 0:
                print(f"[{model_type}] Ep {e+1}/{EPISODES_PER_MODEL} | Score: {total_reward:.1f} | Eps: {agent.epsilon:.2f} | Loss: {avg_loss:.2f}")

    except KeyboardInterrupt:
        print(f"\n🛑 {model_type} 학습 강제 중단됨!")

    finally:
        # 학습 완료(혹은 중단) 시 모델 저장 및 리소스 정리
        env.close()
        torch.save(agent.model.state_dict(), model_filename)
        print(f"✨ Finished & Saved: {model_type}")

        # [중요] 다음 모델을 위해 메모리 정리
        del agent
        del env
        torch.cuda.empty_cache()
        gc.collect()

    return True # 정상 완료

def main():
    print("📢 전체 배치 학습을 시작합니다.")
    print(f"대상 모델: {MODELS_TO_TRAIN}")
    print(f"모델당 에피소드: {EPISODES_PER_MODEL}")
    print(f"중간 저장 빈도: {CHECKPOINT_FREQ} 에피소드")

    for model_name in MODELS_TO_TRAIN:
        success = train_single_model(model_name)
        if not success:
            print("❌ 사용자에 의해 전체 프로그램이 종료되었습니다.")
            break

    print("\n🎉 모든 학습 스케줄이 완료되었습니다! 퇴근하셔도 좋습니다.")

if __name__ == "__main__":
    main()