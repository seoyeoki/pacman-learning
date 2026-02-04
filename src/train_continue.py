import torch
import numpy as np
import os
import random
from collections import deque
from pacman_env import PacmanEnv, WALL
from cnn_model_agent.cnn_ddqn_agent import CNNDDQNAgent
import csv
from datetime import datetime

try:
    from pacman_env import DX, DY
except ImportError:
    DX = [-1, 1, 0, 0]
    DY = [0, 0, -1, 1]

# =========================================================
# [설정] 안전장치 학습 파라미터
# =========================================================
LOAD_MODEL_PATH = "../trained_pth/pacman_cnn_ddqn.pth"
# 재학습된 파일이 있다면 그걸 이어서 학습 (없으면 원본 로드)
if os.path.exists("../trained_pth/pacman_cnn_ddqn_retrained.pth"):
    LOAD_MODEL_PATH = "../trained_pth/pacman_cnn_ddqn_retrained.pth"

SAVE_MODEL_PATH = "../trained_pth/pacman_cnn_ddqn_safe.pth" # 안전장치 학습 모델

ADDITIONAL_EPISODES = 5000   # 5천 번이면 충분할 듯
START_EPSILON = 0.2          # 20%만 탐험 (안전장치가 도와주므로 낮아도 됨)
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.999
# =========================================================

def get_one_hot_state(grid, pacman_pos, ghosts):
    state = np.zeros((5, 20, 20), dtype=np.float32)
    state[0] = (grid == 0)
    state[1] = (grid == 1)
    state[4] = (grid == 4)
    pr, pc = pacman_pos
    state[2][pr, pc] = 1.0
    for gr, gc in ghosts:
        state[3][gr, gc] = 1.0
    return state

def main():
    env = PacmanEnv()
    agent = CNNDDQNAgent(action_size=4)

    # 모델 로드
    if os.path.exists(LOAD_MODEL_PATH):
        print(f"📂 모델 로드 중: {LOAD_MODEL_PATH}")
        try:
            agent.model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location='cpu'))
            agent.target_model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location='cpu'))
            print("✅ 모델 로드 완료! 안전장치 학습을 시작합니다.")
        except Exception as e:
            print(f"⚠️ 모델 로드 실패: {e}")
            return
    else:
        print(f"⚠️ 모델 파일이 없습니다. 처음부터 시작합니다.")

    agent.epsilon = START_EPSILON

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"../train_result/safe_train_log_{current_time}.csv"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Score', 'Steps', 'Epsilon', 'Avg_Loss', 'Wall_Hits', 'Coins'])

    print(f"🚀 안전장치(Safety) 학습 시작 (목표: {ADDITIONAL_EPISODES} 에피소드)")

    for e in range(1, ADDITIONAL_EPISODES + 1):
        env.reset()

        frame_stack = deque(maxlen=4)
        init_frame = get_one_hot_state(env.grid, env.pacman_pos, env.ghosts)
        for _ in range(4):
            frame_stack.append(init_frame)

        state = np.concatenate(frame_stack, axis=0)

        done = False
        score = 0
        step = 0
        losses = []

        while not done:
            # 1. AI의 원래 생각
            original_action = agent.get_action(state)
            final_action = original_action

            # ---------------------------------------------------------
            # 🛡️ [안전장치] 벽으로 가려 하면 강제로 교정 (Teacher Forcing)
            # ---------------------------------------------------------
            pr, pc = env.pacman_pos
            dr, dc = DX[final_action], DY[final_action]
            nr, nc = pr + dr, pc + dc

            # 벽이거나 맵 밖이면?
            if not (0 <= nr < 20 and 0 <= nc < 20) or env.grid[nr, nc] == WALL:
                # 갈 수 있는 곳 찾기
                legal_actions = []
                for i in range(4):
                    ldr, ldc = DX[i], DY[i]
                    lnr, lnc = pr + ldr, pc + ldc
                    if 0 <= lnr < 20 and 0 <= lnc < 20 and env.grid[lnr, lnc] != WALL:
                        legal_actions.append(i)

                if legal_actions:
                    # 안전한 곳 중 하나로 강제 변경!
                    # (여기서 AI는 "아, 내가 원래 이쪽으로 가려 했었지?"라고 착각하게 됨)
                    final_action = random.choice(legal_actions)
            # ---------------------------------------------------------

            next_grid, reward, done, info = env.step(final_action)

            next_frame = get_one_hot_state(next_grid, env.pacman_pos, env.ghosts)
            frame_stack.append(next_frame)
            next_state = np.concatenate(frame_stack, axis=0)

            # [중요] 메모리에는 '보정된 행동(final_action)'을 저장해야 함!
            # 그래야 AI가 "이 상황에선 이게 정답이구나"라고 배움.
            agent.remember(state, final_action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None: losses.append(loss)

            state = next_state
            score += reward
            step += 1

            if done:
                agent.update_target_network()

                # 10판마다 로그 출력
                if e % 10 == 0:
                    avg_loss = np.mean(losses) if losses else 0
                    print(f"Ep {e} | Score: {score:.1f} | Wall: {info['wall_hits']} (Fixed) | Coins: {info.get('coins_eaten', 0)} | Eps: {agent.epsilon:.3f}")

                    with open(log_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([e, score, step, agent.epsilon, avg_loss, info['wall_hits'], info.get('coins_eaten', 0)])

        if agent.epsilon > MIN_EPSILON:
            agent.epsilon *= EPSILON_DECAY

        if e % 1000 == 0:
            torch.save(agent.model.state_dict(), SAVE_MODEL_PATH)
            print(f"💾 안전장치 모델 저장: {SAVE_MODEL_PATH}")

    torch.save(agent.model.state_dict(), SAVE_MODEL_PATH)
    print("🎉 학습 완료! 이제 AI는 벽을 피하는 법을 몸으로 익혔습니다.")

if __name__ == "__main__":
    main()