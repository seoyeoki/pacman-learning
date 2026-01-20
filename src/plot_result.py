import pandas as pd
import matplotlib.pyplot as plt

def plot_training_data(filename='training_log.csv'):
    try:
        # CSV 파일 읽기
        data = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"❌ '{filename}' 파일이 없습니다. train_dqn.py를 먼저 실행하세요.")
        return

    # 이동 평균 (Moving Average) 계산 - 그래프를 부드럽게 보기 위함
    # window=50 : 최근 50개 데이터의 평균을 선으로 그립니다.
    window_size = 50
    data['Score_MA'] = data['Score'].rolling(window=window_size).mean()
    data['Loss_MA'] = data['Avg_Loss'].rolling(window=window_size).mean()

    # 그래프 그리기 (2개 화면 분할)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # 1. 점수 그래프 (Score)
    ax[0].plot(data['Episode'], data['Score'], label='Raw Score', color='lightgray', alpha=0.5)
    ax[0].plot(data['Episode'], data['Score_MA'], label=f'Moving Avg ({window_size})', color='blue', linewidth=2)
    ax[0].set_title('Training Score (Reward)')
    ax[0].set_ylabel('Score')
    ax[0].legend()
    ax[0].grid(True)

    # 2. 오차 그래프 (Loss)
    ax[1].plot(data['Episode'], data['Avg_Loss'], label='Raw Loss', color='lightcoral', alpha=0.5)
    ax[1].plot(data['Episode'], data['Loss_MA'], label=f'Moving Avg ({window_size})', color='red', linewidth=2)
    ax[1].set_title('Training Loss')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show() # 창 띄우기
    # plt.savefig('training_result.png') # 파일로 저장하려면 주석 해제

if __name__ == "__main__":
    plot_training_data()