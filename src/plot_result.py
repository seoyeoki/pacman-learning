import pandas as pd
import matplotlib.pyplot as plt

def plot_training_data(filename='training_log.csv'):
    try:
        data = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"❌ '{filename}' 파일이 없습니다.")
        return

    # 이동 평균 (부드럽게 보기)
    window_size = 50
    data['Score_MA'] = data['Score'].rolling(window=window_size).mean()
    data['Wall_MA'] = data['Wall_Hits'].rolling(window=window_size).mean()
    data['Coins_MA'] = data['Coins'].rolling(window=window_size).mean()

    # 2x2 화면 분할
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 점수 (Score) - 우상향해야 함 ↗️
    ax[0, 0].plot(data['Episode'], data['Score'], color='lightgray', alpha=0.5)
    ax[0, 0].plot(data['Episode'], data['Score_MA'], color='blue', linewidth=2)
    ax[0, 0].set_title('Score (Higher is Better)')
    ax[0, 0].grid(True)

    # 2. 오차 (Loss) - 0으로 수렴하거나 일정하게 유지 ↘️ or ➡️
    ax[0, 1].plot(data['Episode'], data['Avg_Loss'], color='lightcoral', alpha=0.5)
    # Loss는 변동이 심해 이동평균만 보는 게 나을 수 있음
    ax[0, 1].plot(data['Episode'], data['Avg_Loss'].rolling(window=50).mean(), color='red', linewidth=2)
    ax[0, 1].set_title('Loss (Stability Check)')
    ax[0, 1].grid(True)

    # 3. 벽 충돌 (Wall Hits) - 우하향해야 함 ↘️ (가장 중요!)
    ax[1, 0].plot(data['Episode'], data['Wall_Hits'], color='lightgray', alpha=0.5)
    ax[1, 0].plot(data['Episode'], data['Wall_MA'], color='green', linewidth=2)
    ax[1, 0].set_title('Wall Hits (Lower is Better)')
    ax[1, 0].grid(True)

    # 4. 코인 획득 (Coins) - 우상향해야 함 ↗️
    ax[1, 1].plot(data['Episode'], data['Coins'], color='lightgray', alpha=0.5)
    ax[1, 1].plot(data['Episode'], data['Coins_MA'], color='orange', linewidth=2)
    ax[1, 1].set_title('Coins Eaten (Higher is Better)')
    ax[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_data()