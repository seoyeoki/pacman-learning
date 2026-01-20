import pandas as pd
import matplotlib.pyplot as plt
import os

# =================================================================
# [설정] 그래프 모드 선택
# 1. 보고 싶은 모델 이름 ("DQN", "DDQN", "DUELING")
MODEL_TYPE = "DDQN"

# 2. 비교 모드 (True로 하면 DQN과 DDQN을 한 그래프에 겹쳐서 비교합니다!)
COMPARE_MODE = False
# =================================================================

def plot_single_model(model_name):
    """하나의 모델에 대해 자세한 4분할 그래프 그리기"""
    filename = f"log_{model_name.lower()}.csv"

    if not os.path.exists(filename):
        print(f"❌ '{filename}' 파일이 없습니다. 먼저 학습을 진행하세요.")
        return

    print(f"📊 {model_name} 학습 데이터 로딩 중... ({filename})")
    data = pd.read_csv(filename)

    # 이동 평균 (Moving Average)
    window_size = 50
    data['Score_MA'] = data['Score'].rolling(window=window_size).mean()
    data['Wall_MA'] = data['Wall_Hits'].rolling(window=window_size).mean()
    data['Coins_MA'] = data['Coins'].rolling(window=window_size).mean()

    # Loss는 값이 없을 수도 있으므로 예외 처리
    if 'Avg_Loss' in data.columns:
        data['Loss_MA'] = data['Avg_Loss'].rolling(window=window_size).mean()
    else:
        data['Avg_Loss'] = 0
        data['Loss_MA'] = 0

    # 그래프 그리기
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Training Result: {model_name}", fontsize=16)

    # 1. 점수 (Score)
    ax[0, 0].plot(data['Episode'], data['Score'], color='lightgray', alpha=0.5, label='Raw')
    ax[0, 0].plot(data['Episode'], data['Score_MA'], color='blue', linewidth=2, label='Moving Avg')
    ax[0, 0].set_title('Score (Higher is Better)')
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    # 2. 오차 (Loss)
    ax[0, 1].plot(data['Episode'], data['Avg_Loss'], color='lightcoral', alpha=0.3)
    ax[0, 1].plot(data['Episode'], data['Loss_MA'], color='red', linewidth=2)
    ax[0, 1].set_title('Loss (Stability Check)')
    ax[0, 1].grid(True)

    # 3. 벽 충돌 (Wall Hits)
    ax[1, 0].plot(data['Episode'], data['Wall_Hits'], color='lightgray', alpha=0.5)
    ax[1, 0].plot(data['Episode'], data['Wall_MA'], color='green', linewidth=2)
    ax[1, 0].set_title('Wall Hits (Lower is Better)')
    ax[1, 0].grid(True)

    # 4. 코인 획득 (Coins)
    ax[1, 1].plot(data['Episode'], data['Coins'], color='lightgray', alpha=0.5)
    ax[1, 1].plot(data['Episode'], data['Coins_MA'], color='orange', linewidth=2)
    ax[1, 1].set_title('Coins Eaten (Higher is Better)')
    ax[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_comparison():
    """여러 모델의 성능을 한 화면에서 비교 (A/B Test)"""
    models = ["DQN", "DDQN", "DUELING"]
    colors = {"DQN": "blue", "DDQN": "red", "DUELING": "green"}

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison (Moving Average)", fontsize=16)

    found_any = False

    for model in models:
        filename = f"log_{model.lower()}.csv"
        if not os.path.exists(filename):
            continue

        found_any = True
        data = pd.read_csv(filename)
        window_size = 50

        # 데이터가 너무 적으면 window 사이즈 조절
        if len(data) < window_size:
            real_window = 1
        else:
            real_window = window_size

        # 점수 비교
        score_ma = data['Score'].rolling(window=real_window).mean()
        ax[0, 0].plot(data['Episode'], score_ma, label=model, color=colors[model], linewidth=2)

        # Loss 비교
        loss_ma = data['Avg_Loss'].rolling(window=real_window).mean()
        ax[0, 1].plot(data['Episode'], loss_ma, label=model, color=colors[model], linewidth=2)

        # 벽 충돌 비교
        wall_ma = data['Wall_Hits'].rolling(window=real_window).mean()
        ax[1, 0].plot(data['Episode'], wall_ma, label=model, color=colors[model], linewidth=2)

        # 코인 비교
        coin_ma = data['Coins'].rolling(window=real_window).mean()
        ax[1, 1].plot(data['Episode'], coin_ma, label=model, color=colors[model], linewidth=2)

    if not found_any:
        print("❌ 비교할 로그 파일이 하나도 없습니다.")
        return

    # 그래프 세팅
    ax[0, 0].set_title('Score (Higher is Better)')
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    ax[0, 1].set_title('Loss')
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    ax[1, 0].set_title('Wall Hits (Lower is Better)')
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    ax[1, 1].set_title('Coins Eaten (Higher is Better)')
    ax[1, 1].legend()
    ax[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if COMPARE_MODE:
        print("⚔️ 비교 모드 실행: 모든 로그 파일을 불러옵니다...")
        plot_comparison()
    else:
        print(f"🔍 단일 모드 실행: {MODEL_TYPE} 분석 중...")
        plot_single_model(MODEL_TYPE)