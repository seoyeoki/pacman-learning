import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 하이퍼파라미터 ---
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.99
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        # [수정] 입력이 2000개로 늘어났으니, 중간 레이어도 좀 키워줍니다 (128 -> 256)
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

        # [핵심 수정] 네트워크를 2개 만듭니다. (행동대장 & 정답지 선생님)
        self.model = QNetwork(state_size, action_size).to(device)
        self.target_model = QNetwork(state_size, action_size).to(device)

        # 처음에는 두 네트워크가 똑같게 설정
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # 타겟 모델은 학습(Backprop)하지 않음!

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # 1. 현재 상태의 Q값 (모델이 예측)
        current_q = self.model(states).gather(1, actions)

        # 2. 다음 상태의 최대 Q값 (타겟 모델이 예측) [핵심 변경]
        # self.model 대신 self.target_model을 사용해야 학습이 안정됩니다.
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (GAMMA * max_next_q * (1 - dones))

        # 3. 오차 계산 및 업데이트
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """[추가] 타겟 네트워크를 현재 모델과 똑같이 동기화"""
        self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        if self.epsilon > EPSILON_END:
            self.epsilon *= self.epsilon_decay