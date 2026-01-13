import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 하이퍼파라미터 (튜닝 가능) ---
BATCH_SIZE = 64         # 한 번에 학습할 데이터 양
LR = 0.001              # 학습률 (Learning Rate)
GAMMA = 0.99            # 할인율 (미래 보상을 얼마나 중요하게 여길지)
MEMORY_SIZE = 10000     # 기억 용량 (Replay Buffer)
EPSILON_START = 1.0     # 초기 탐험 확률 (100% 랜덤 행동)
EPSILON_END = 0.01      # 최소 탐험 확률 (1% 랜덤)
EPSILON_DECAY = 0.995   # 탐험 확률 감소 비율

# --- GPU 사용 설정 (가능하면 GPU, 아니면 CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Deep Q-Network 구조 정의"""
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        # 20x20 그리드(400개)를 입력으로 받아 4개의 행동 가치를 출력
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # 모델 생성 (예측용, 타겟용 2개 운영이 정석이나 여기선 간소화)
        self.model = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        # 리플레이 메모리 (경험 저장소)
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.epsilon = EPSILON_START

    def get_action(self, state):
        """엡실론-그리디 정책에 따라 행동 선택"""
        # 탐험: 랜덤 행동
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))

        # 활용: 모델이 생각하는 최적 행동
        state = torch.FloatTensor(state).unsqueeze(0).to(device) # (1, 400) 형태로 변환
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """경험(데이터) 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """저장된 경험을 랜덤하게 뽑아서 학습 (Replay)"""
        if len(self.memory) < BATCH_SIZE:
            return

        # 랜덤 배치 추출
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 텐서 변환
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # 현재 상태의 Q값 예측
        current_q = self.model(states).gather(1, actions)

        # 다음 상태의 최대 Q값 예측 (타겟 계산)
        with torch.no_grad():
            max_next_q = self.model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (GAMMA * max_next_q * (1 - dones))

        # 손실 계산 및 업데이트
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        """탐험 비율 감소"""
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY