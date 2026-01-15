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

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Deep Q-Network 구조 정의"""
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
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
    """[핵심] 여기에 DQNAgent 클래스가 정의되어 있어야 합니다!"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # 모델 생성
        self.model = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        # 리플레이 메모리
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
        """학습 (Experience Replay)"""
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_q = self.model(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (GAMMA * max_next_q * (1 - dones))

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        """탐험률 감소"""
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY