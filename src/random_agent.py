import random

class RandomAgent:
    """
    아무런 학습 없이 무작위로 행동하는 에이전트.
    성능 비교의 기준점(Baseline)이 됩니다.
    """
    def __init__(self, action_size):
        self.action_size = action_size

    # state를 받긴 하지만, 보지 않고 랜덤 행동을 반환합니다.
    def get_action(self, state):
        return random.choice(range(self.action_size))