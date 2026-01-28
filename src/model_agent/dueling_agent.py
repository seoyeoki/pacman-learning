import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ==========================================
# [설정] 하이퍼파라미터
# ==========================================
learning_rate = 0.00025
gamma = 0.99
buffer_limit = 50000
batch_size = 64
epsilon_init = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(np.array(s_lst), dtype=torch.float).to(device), \
            torch.tensor(np.array(a_lst), dtype=torch.long).to(device), \
            torch.tensor(np.array(r_lst), dtype=torch.float).to(device), \
            torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(device), \
            torch.tensor(np.array(done_mask_lst), dtype=torch.float).to(device)

    def size(self):
        return len(self.buffer)

    # [수정 1] train.py의 len(agent.memory) 호출에 대응
    def __len__(self):
        return len(self.buffer)

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()

        # 1. Feature Layer (공통)
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 2. Value Stream (상태 가치)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 3. Advantage Stream (행동 이점)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        feature = self.feature_layer(x)
        value = self.value_stream(feature)
        advantage = self.advantage_stream(feature)

        # Q = V + (A - mean(A))
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value

class DuelingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # 네트워크 초기화
        self.model = DuelingQNetwork(state_size, action_size).to(device)
        self.target_model = DuelingQNetwork(state_size, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer()
        self.epsilon = epsilon_init

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return q_values.argmax().item()

    # [수정 2] train.py의 agent.remember(...) 호출에 대응
    def remember(self, state, action, reward, next_state, done):
        done_mask = 0.0 if done else 1.0
        self.memory.put((state, action, reward, next_state, done_mask))

    # [수정 3] train.py의 agent.train_step() 호출에 대응
    def train_step(self):
        # 메모리가 부족하면 학습 스킵 (train.py에서 체크하지만 이중 안전장치)
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, done_masks = self.memory.sample(batch_size)

        # 현재 상태 Q값
        q_out = self.model(states).gather(1, actions)

        # 타겟 Q값 (Double DQN 방식 적용)
        # Main Network로 행동 선택 -> Target Network로 가치 평가
        max_actions = self.model(next_states).argmax(dim=1, keepdim=True)
        target_q = self.target_model(next_states).gather(1, max_actions)

        expected_q = rewards + (gamma * target_q * done_masks)

        loss = nn.MSELoss()(q_out, expected_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # 기울기 폭발 방지
        self.optimizer.step()

        return loss.item()

    # [수정 4] train.py의 agent.update_target_network() 호출에 대응
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # [수정 5] train.py의 agent.update_epsilon() 호출에 대응
    def update_epsilon(self):
        self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)