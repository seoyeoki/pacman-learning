import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# CNN 하이퍼파라미터
# 맵이 랜덤하므로 다양한 상황을 기억하기 위해 버퍼를 크게 유지
BATCH_SIZE = 64
LR = 0.00025
GAMMA = 0.99
BUFFER_LIMIT = 50000
EPSILON_DECAY = 0.9995

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNDuelingNetwork(nn.Module):
    def __init__(self, action_size):
        super(CNNDuelingNetwork, self).__init__()

        # [CNN Layer] 이미지 처럼 특징 추출
        # 입력: (Batch, 5, 20, 20) -> 5채널 (Empty, Wall, Pacman, Ghost, Coin)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Flatten 후의 크기 계산: 20x20 이미지는 conv를 거쳐도 padding=1이라 20x20 유지
        # 64채널 * 20 * 20 = 25600
        self.flatten_size = 64 * 20 * 20

        # Dueling 구조 (Value & Advantage)
        self.fc_val = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1) # V(s)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size) # A(s, a)
        )

    def forward(self, x):
        # x shape: (Batch, 5, 20, 20)
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size(0), -1) # Flatten

        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)

        # Q = V + A - mean(A)
        return val + (adv - adv.mean(dim=1, keepdim=True))

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_LIMIT)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        batch = random.sample(self.buffer, n)
        s, a, r, s_next, done = zip(*batch)
        return s, a, r, s_next, done

class CNNDuelingAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.learning_rate = LR
        self.epsilon = 1.0
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = 0.01
        self.train_start = 1000

        self.memory = ReplayBuffer()

        self.model = CNNDuelingNetwork(action_size).to(device)
        self.target_model = CNNDuelingNetwork(action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state):
        # state shape: (5, 20, 20) -> 모델 입력은 (1, 5, 20, 20)이어야 함
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.put((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory.buffer) < self.train_start:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)

        # Dueling DQN Logic
        curr_q = self.model(states).gather(1, actions)

        # Double DQN Target
        next_actions = self.model(next_states).argmax(1, keepdim=True)
        target_q = self.target_model(next_states).gather(1, next_actions)
        expected_q = rewards + (GAMMA * target_q * (1 - dones))

        loss = nn.MSELoss()(curr_q, expected_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()