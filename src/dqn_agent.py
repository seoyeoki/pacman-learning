import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 하이퍼파라미터 (5,000판 기준) ---
BATCH_SIZE = 64
LR = 0.0005
GAMMA = 0.99
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.999

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Deep Q-Network 구조 정의"""
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        # 20x20x5 입력에 맞춰 256 뉴런으로 시작
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

        # 1. 메인 모델 (행동대장)
        self.model = QNetwork(state_size, action_size).to(device)

        # 2. 타겟 모델 (채점관)
        self.target_model = QNetwork(state_size, action_size).to(device)

        # 초기화: 타겟 모델을 메인 모델과 똑같이 만듦
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # 타겟 모델은 학습하지 않음

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
        """학습 (Double DQN + Experience Replay)"""
        if len(self.memory) < BATCH_SIZE:
            return None # 데이터가 부족하면 학습 안 함 (Loss 없음)

        # 랜덤 배치 추출
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # 1. 현재 상태에서의 예측값 (Main Model)
        current_q = self.model(states).gather(1, actions)

        # ---------------------------------------------------------
        # [Double DQN 핵심 로직]
        # ---------------------------------------------------------
        # A. 행동 선택: '메인 모델'이 다음 상태에서 제일 좋은 행동을 고름 (argmax)
        next_actions = self.model(next_states).argmax(1).unsqueeze(1)

        # B. 점수 평가: '타겟 모델'이 그 행동의 점수를 매김 (gather)
        #    detach()는 타겟 모델로 역전파가 흐르지 않게 차단하는 역할
        with torch.no_grad():
            max_next_q = self.target_model(next_states).gather(1, next_actions)

            # C. 정답(Target Q) 계산
            target_q = rewards + (GAMMA * max_next_q * (1 - dones))
        # ---------------------------------------------------------

        # 오차 계산 및 업데이트
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 로그 기록을 위해 Loss 값 반환
        return loss.item()

    def update_target_network(self):
        """타겟 모델 동기화"""
        self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        """탐험률 감소"""
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY