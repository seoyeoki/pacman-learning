import random

class RandomAgent:
    """
    아무 학습 없이 무작위로 행동하는 에이전트
    비교군(Baseline)으로 사용됩니다.
    """
    def __init__(self, action_size):
        self.action_size = action_size

    def get_action(self, state):
        # 상태(state)는 보지 않고 무조건 랜덤 행동 반환
        return random.randrange(self.action_size)