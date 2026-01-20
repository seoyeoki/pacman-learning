import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# --- 매개변수 ---

BATCH_SIZE = 64         # 한 번에 학습할 데이터 양 (RTX 4060이므로 128도 가능)
LR = 0.0005             # 학습률 (0.001은 가끔 너무 튀어서, 조금 더 안정적인 0.0005 추천)
GAMMA = 0.99            # 할인율 (미래 보상을 얼마나 중요하게 볼지, 0.99가 국룰)
MEMORY_SIZE = 10000     # 리플레이 버퍼 크기 (최근 1만 개의 경험만 기억)

# 탐험(Exploration) 관련 설정
EPSILON_START = 1.0     # 100% 랜덤으로 시작
EPSILON_END = 0.01      # 최소 1%는 계속 탐험하게 둠
EPSILON_DECAY = 0.999   # 감소 속도 (이 값이 클수록 천천히 줄어듦)

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Deep Q-Network 구조 정의"""
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        # 입력이 2000개(20x20x5)로 늘어났으니, 첫 레이어도 256으로 확장
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # [핵심] 네트워크 2개 생성 (행동대장 & 정답지 선생님)
        self.model = QNetwork(state_size, action_size).to(device)
        self.target_model = QNetwork(state_size, action_size).to(device)

        # 처음엔 둘을 똑같이 동기화
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # 타겟 모델은 학습(역전파) 안 함

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.epsilon = EPSILON_START

    def get_action(self, state):
        """행동 선택 (Epsilon-Greedy)"""
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """학습 (Experience Replay + Target Network)"""
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # 1. 현재 예측값 (Main Model)
        current_q = self.model(states).gather(1, actions)

        # 2. 정답값 (Target Model 사용) -> 학습 안정화의 핵심!
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (GAMMA * max_next_q * (1 - dones))

        # 3. 오차 계산 및 업데이트
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """타겟 모델을 현재 모델과 동기화"""
        self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        """탐험률 감소"""
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY