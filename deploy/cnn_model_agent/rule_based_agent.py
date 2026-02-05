import numpy as np
import random
from collections import deque

class RuleBasedAgent:
    def __init__(self, action_size=4, moves=None):
        self.action_size = action_size

        # [수정] 외부에서 정확한 이동 규칙(DX, DY)을 받아옴
        if moves is not None:
            self.moves = moves
        else:
            # 기본값 (환경과 다를 수 있음 -> 위험!)
            self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_action(self, state):
        """
        state shape: (20, 20, 20) -> 프레임 스태킹 대응
        """
        if state.shape[0] == 20:
            current_state = state[-5:, :, :]
        else:
            current_state = state

        rows, cols = current_state.shape[1], current_state.shape[2]
        walls = current_state[1]
        pacman_layer = current_state[2]
        ghosts = current_state[3]
        coins = current_state[4]

        pacman_pos = np.argwhere(pacman_layer == 1)
        if len(pacman_pos) == 0: return random.randint(0, 3)
        pr, pc = pacman_pos[0]

        # 1. 물리적 이동 가능성 체크 (Legal Moves)
        legal_actions = []
        for action_idx, (dr, dc) in enumerate(self.moves):
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # 벽이 아니면 OK (state[1]이 정확해야 함)
                if walls[nr, nc] == 0:
                    legal_actions.append(action_idx)

        if not legal_actions: return 0

        # 2. 안전한 이동 (Safe Moves)
        safe_actions = []
        for action in legal_actions:
            dr, dc = self.moves[action]
            nr, nc = pr + dr, pc + dc
            if ghosts[nr, nc] == 0:
                safe_actions.append(action)

        random.shuffle(safe_actions)

        # 3. BFS 탐색
        if not safe_actions: return random.choice(legal_actions)

        queue = deque()
        visited = set()
        visited.add((pr, pc))

        for action in safe_actions:
            dr, dc = self.moves[action]
            nr, nc = pr + dr, pc + dc
            queue.append((nr, nc, action))
            visited.add((nr, nc))

        best_action = None

        while queue:
            r, c, first_action = queue.popleft()
            if coins[r, c] == 1:
                best_action = first_action
                break

            check_dirs = list(self.moves)
            random.shuffle(check_dirs)

            for dr, dc in check_dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in visited and walls[nr, nc] == 0 and ghosts[nr, nc] == 0:
                        visited.add((nr, nc))
                        queue.append((nr, nc, first_action))

        return best_action if best_action is not None else random.choice(safe_actions)