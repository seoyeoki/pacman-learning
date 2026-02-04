import pygame
import torch
import time
import numpy as np
import csv
import os
from collections import deque
from datetime import datetime
from pacman_env import PacmanEnv, WALL, EMPTY, PACMAN, GHOST, COIN

try:
    from pacman_env import DX, DY
except ImportError:
    print("⚠️ Warning: Could not import DX, DY. Using defaults.")
    DX = [-1, 1, 0, 0]
    DY = [0, 0, -1, 1]

# =================================================================
MODEL_TYPE = "CNN_DDQN"
NUM_TEST_EPISODES = 10
# =================================================================

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_FILENAME = f"../test_result/test_summary_{MODEL_TYPE}_{current_time}.csv"

model_filename = None

if MODEL_TYPE == "CNN_DQN":
    from cnn_model_agent.cnn_dqn_agent import CNNDQNAgent as AgentClass
    model_filename = "../trained_pth/pacman_cnn_dqn.pth"
elif MODEL_TYPE == "CNN_DDQN":
    from cnn_model_agent.cnn_ddqn_agent import CNNDDQNAgent as AgentClass

    # [수정] 재학습된 모델이 있으면 우선 사용하되, 없으면 기본 모델 사용
    retrained_path = "../trained_pth/pacman_cnn_ddqn_retrained.pth"
    base_path = "../trained_pth/pacman_cnn_ddqn.pth"

    if os.path.exists(retrained_path):
        model_filename = retrained_path
        print(f"✨ Found RETRAINED Model: {model_filename}")
    else:
        model_filename = base_path
        print(f"ℹ️ Using Base Model: {model_filename}")

elif MODEL_TYPE == "CNN_DUELING":
    from cnn_model_agent.cnn_dueling_agent import CNNDuelingAgent as AgentClass
    model_filename = "../trained_pth/pacman_cnn_dueling.pth"
elif MODEL_TYPE == "RANDOM":
    from cnn_model_agent.random_agent import RandomAgent as AgentClass
    model_filename = None
elif MODEL_TYPE == "RULE_BASED":
    from cnn_model_agent.rule_based_agent import RuleBasedAgent as AgentClass
    model_filename = None
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

# 하드코딩된 숫자 대신 실제 상수를 사용하여 상태 생성
def get_one_hot_state(grid, pacman_pos, ghosts):
    state = np.zeros((5, 20, 20), dtype=np.float32)
    state[0] = (grid == EMPTY)
    state[1] = (grid == WALL)
    state[4] = (grid == COIN)
    pr, pc = pacman_pos
    state[2][pr, pc] = 1.0
    for gr, gc in ghosts:
        state[3][gr, gc] = 1.0
    return state

def run_episode(env, agent, episode_idx):
    env.reset()

    # 프레임 스태킹
    frame_stack = deque(maxlen=4)
    init_frame = get_one_hot_state(env.grid, env.pacman_pos, env.ghosts)
    for _ in range(4):
        frame_stack.append(init_frame)

    done = False
    total_reward = 0
    step = 0
    action_counts = [0, 0, 0, 0]

    pygame.display.set_caption(f"{MODEL_TYPE} Test - Ep {episode_idx}")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        stacked_state = np.concatenate(frame_stack, axis=0)

        # 1. AI의 의견 (아마도 계속 왼쪽이라고 할 것임)
        action = agent.get_action(stacked_state)

        # =================================================================
        # 🛡️ [안전장치] AI가 벽으로 돌진하면 강제로 막음 (Safety Wrapper)
        # =================================================================
        pr, pc = env.pacman_pos
        dr, dc = DX[action], DY[action]
        nr, nc = pr + dr, pc + dc

        # "거긴 벽이야! 못 가!" -> 갈 수 있는 다른 길 찾기
        if not (0 <= nr < 20 and 0 <= nc < 20) or env.grid[nr, nc] == WALL:
            # 갈 수 있는(벽이 아닌) 모든 방향 조사
            legal_actions = []
            for i in range(4):
                ldr, ldc = DX[i], DY[i]
                lnr, lnc = pr + ldr, pc + ldc
                if 0 <= lnr < 20 and 0 <= lnc < 20 and env.grid[lnr, lnc] != WALL:
                    legal_actions.append(i)

            if legal_actions:
                # 안전한 길 중 하나로 강제 변경 (랜덤)
                # 이렇게 하면 '왼쪽'이 막혔을 때 다른 곳으로 튕겨 나옵니다.
                action = np.random.choice(legal_actions)
        # =================================================================

        if 0 <= action < 4:
            action_counts[action] += 1

        next_grid, reward, done, info = env.step(action)

        next_frame = get_one_hot_state(next_grid, env.pacman_pos, env.ghosts)
        frame_stack.append(next_frame)

        total_reward += reward
        step += 1
        env.render()

    return {
        'episode': episode_idx,
        'score': total_reward,
        'steps': step,
        'wall_hits': info['wall_hits'],
        'coins': info['coins_eaten'],
        'actions': action_counts
    }

def main():
    env = PacmanEnv()
    action_size = 4

    # RuleBasedAgent만 moves 인자를 받으므로 분기 처리
    if MODEL_TYPE == "RULE_BASED":
        real_moves = list(zip(DX, DY))
        agent = AgentClass(action_size, moves=real_moves)
    else:
        agent = AgentClass(action_size)

    if model_filename is not None:
        print(f"📂 Loading Model from: {model_filename}")
        try:
            agent.model.load_state_dict(torch.load(model_filename, map_location='cpu', weights_only=True))
            agent.epsilon = 0.0 # 테스트니 탐험 끄기
            print("✅ Model Loaded Successfully!")
        except FileNotFoundError:
            print(f"❌ Error: Model file not found at {model_filename}")
            return
        except RuntimeError as e:
            print(f"❌ Model Shape Mismatch: {e}")
            return
    else:
        print(f"🤖 {MODEL_TYPE} Agent Selected")

    history = []
    print(f"\n🚀 Start Testing ({NUM_TEST_EPISODES} Episodes) ---")

    for i in range(1, NUM_TEST_EPISODES + 1):
        res = run_episode(env, agent, i)
        history.append(res)
        acts = res['actions']
        print(f"Ep {i} | Score: {res['score']:.1f} | Wall: {res['wall_hits']} | Coins: {res['coins']} | Move: U{acts[0]} D{acts[1]} L{acts[2]} R{acts[3]}")

    scores = [h['score'] for h in history]
    walls = [h['wall_hits'] for h in history]
    coins = [h['coins'] for h in history]
    steps = [h['steps'] for h in history]
    total_actions = np.sum([h['actions'] for h in history], axis=0)

    print("\n" + "="*50)
    print(f"   📊 [ {MODEL_TYPE} ] 최종 성적표 (총 {NUM_TEST_EPISODES}회)")
    print("="*50)
    print(f"   🏆 평균 점수 : {np.mean(scores):.2f}")
    print(f"   🪙 평균 코인 : {np.mean(coins):.1f}")
    print(f"   💥 평균 충돌 : {np.mean(walls):.1f}")
    print(f"   🦶 평균 스텝 : {np.mean(steps):.1f}")
    print("-" * 50)
    print(f"   ⬆️  UP    : {total_actions[0]}회")
    print(f"   ⬇️  DOWN  : {total_actions[1]}회")
    print(f"   ⬅️  LEFT  : {total_actions[2]}회")
    print(f"   ➡️  RIGHT : {total_actions[3]}회")
    print("-" * 50)

    try:
        with open(RESULT_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Score', 'Wall_Hits', 'Coins', 'Steps', 'Up', 'Down', 'Left', 'Right'])
            for h in history:
                acts = h['actions']
                writer.writerow([h['episode'], h['score'], h['wall_hits'], h['coins'], h['steps'], acts[0], acts[1], acts[2], acts[3]])
        print(f"📁 Log saved to {RESULT_FILENAME}")
    except Exception as e:
        print(f"⚠️ Failed to save CSV: {e}")

if __name__ == "__main__":
    main()