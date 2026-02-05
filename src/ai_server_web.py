from flask import Flask, request, jsonify
import torch
import numpy as np
from collections import deque
from cnn_model_agent.cnn_ddqn_agent import CNNDDQNAgent
import json

app = Flask(__name__)

# ==========================================
# [설정] 모델 경로
MODEL_PATH = "../trained_pth/pacman_cnn_ddqn_safe.pth" # 안전장치 학습된 모델 추천
# ==========================================

# 1. 모델 로드 (서버 시작 시 한 번만 실행)
print("🧠 AI 모델 로딩 중...")
agent = CNNDDQNAgent(action_size=4)
try:
    # CPU 모드로 로드
    agent.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    agent.epsilon = 0.0 # 실전 모드 (탐험 X)
    print("✅ 모델 로드 완료!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    # 실패해도 서버는 켜지게 둠 (디버깅용)

# 프레임 스태킹용 (간단히 전역 변수 사용 - 1:1 시연용)
frame_stack = deque(maxlen=4)

def get_one_hot_state(grid, pacman_pos, ghosts):
    # (기존과 동일한 전처리 함수)
    state = np.zeros((5, 20, 20), dtype=np.float32)
    state[0] = (grid == 0)
    state[1] = (grid == 1)
    state[4] = (grid == 4)
    pr, pc = pacman_pos
    state[2][pr, pc] = 1.0
    for gr, gc in ghosts:
        state[3][gr, gc] = 1.0
    return state

@app.route('/', methods=['POST'])
def predict():
    try:
        # 1. 클라이언트(내 노트북)가 보낸 데이터 받기
        # text 형식이지만 내부엔 JSON이 들어있다고 가정
        if request.is_json:
            data = request.json
        else:
            data = json.loads(request.data)

        grid = np.array(data['grid'])
        pacman_pos = data['pacman']
        ghosts = data['ghosts']
        is_reset = data.get('reset', False)

        # 2. 상태 전처리
        current_frame = get_one_hot_state(grid, pacman_pos, ghosts)

        # 3. 프레임 스태킹 관리
        if is_reset or len(frame_stack) == 0:
            frame_stack.clear()
            for _ in range(4): frame_stack.append(current_frame)
        else:
            frame_stack.append(current_frame)

        # 4. AI 추론
        state = np.concatenate(frame_stack, axis=0)
        action = agent.get_action(state)

        # 5. 결과 반환 (Text 형식)
        return str(action)

    except Exception as e:
        return str(f"Error: {e}"), 500

if __name__ == '__main__':
    # 0.0.0.0으로 열어야 외부 접속 가능
    app.run(host='0.0.0.0', port=30724)