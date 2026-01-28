import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime

# =================================================================
# [설정] 그래프 모드 선택
# 1. 보고 싶은 모델 이름 ("DQN", "DDQN", "DUELING")
MODEL_TYPE = "DDQN"

# 2. 비교 모드 (True로 하면 모든 모델을 한 그래프에 겹쳐서 비교)
COMPARE_MODE = False

# 3. 로그 파일이 저장된 경로
LOG_DIR = "../train_result"
# =================================================================

def get_latest_log_file(model_type):
    """지정된 폴더에서 규칙에 맞는 가장 최근 로그 파일을 찾습니다."""
    if not os.path.exists(LOG_DIR):
        print(f"❌ 폴더를 찾을 수 없습니다: {LOG_DIR}")
        return None

    # [수정] 파일명 규칙 변경: log_ -> train_log_
    # 예시: train_log_dqn_20260121_100000.csv
    file_pattern = f"train_log_{model_type.lower()}_*.csv"
    search_path = os.path.join(LOG_DIR, file_pattern)

    list_of_files = glob.glob(search_path)

    if not list_of_files:
        return None

    # 생성 시간 기준 가장 최근 파일 선택
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def plot_single_model(model_name):
    """하나의 모델에 대해 자세한 4분할 그래프 그리기 및 저장"""
    filename = get_latest_log_file(model_name)

    if filename is None:
        print(f"❌ '{model_name}' 모델의 로그 파일을 '{LOG_DIR}'에서 찾을 수 없습니다.")
        print(f"   (검색 패턴: train_log_{model_name.lower()}_*.csv)")
        return

    print(f"📊 {model_name} 학습 데이터 로딩 중... ({os.path.basename(filename)})")
    try:
        data = pd.read_csv(filename)
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return

    if len(data) < 5:
        print("⚠️ 데이터가 너무 적어 그래프를 그릴 수 없습니다.")
        return

    window_size = max(5, len(data) // 20)

    # 이동 평균 계산
    data['Score_MA'] = data['Score'].rolling(window=window_size).mean()
    data['Wall_MA'] = data['Wall_Hits'].rolling(window=window_size).mean()
    data['Coins_MA'] = data['Coins'].rolling(window=window_size).mean()

    if 'Avg_Loss' in data.columns:
        data['Loss_MA'] = data['Avg_Loss'].rolling(window=window_size).mean()
    else:
        data['Avg_Loss'] = 0
        data['Loss_MA'] = 0

    # 그래프 그리기
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    base_filename = os.path.basename(filename)
    fig.suptitle(f"Training Result: {model_name}\n({base_filename})", fontsize=14)

    # 1. Score
    ax[0, 0].plot(data['Episode'], data['Score'], color='lightgray', alpha=0.5, label='Raw')
    ax[0, 0].plot(data['Episode'], data['Score_MA'], color='blue', linewidth=2, label='Moving Avg')
    ax[0, 0].set_title('Score (Higher is Better)')
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    # 2. Loss
    ax[0, 1].plot(data['Episode'], data['Avg_Loss'], color='lightcoral', alpha=0.3)
    ax[0, 1].plot(data['Episode'], data['Loss_MA'], color='red', linewidth=2)
    ax[0, 1].set_title('Loss (Stability Check)')
    ax[0, 1].grid(True)

    # 3. Wall Hits
    ax[1, 0].plot(data['Episode'], data['Wall_Hits'], color='lightgray', alpha=0.5)
    ax[1, 0].plot(data['Episode'], data['Wall_MA'], color='green', linewidth=2)
    ax[1, 0].set_title('Wall Hits (Lower is Better)')
    ax[1, 0].grid(True)

    # 4. Coins
    ax[1, 1].plot(data['Episode'], data['Coins'], color='lightgray', alpha=0.5)
    ax[1, 1].plot(data['Episode'], data['Coins_MA'], color='orange', linewidth=2)
    ax[1, 1].set_title('Coins Eaten (Higher is Better)')
    ax[1, 1].grid(True)

    plt.tight_layout()

    # [수정] 저장 파일명 규칙 변경: train_log_ -> plot_
    save_path = filename.replace("train_log_", "plot_").replace(".csv", ".png")
    plt.savefig(save_path)
    print(f"💾 그래프 이미지 저장 완료: {save_path}")

    plt.show()

def plot_comparison():
    """여러 모델 비교 그래프 그리기 및 저장"""
    models = ["DQN", "DDQN", "DUELING"]
    colors = {"DQN": "blue", "DDQN": "red", "DUELING": "green"}

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison (Moving Average)", fontsize=16)

    found_any = False

    for model in models:
        filename = get_latest_log_file(model)
        if filename is None:
            continue

        found_any = True
        print(f"📈 {model} 데이터 로드: {os.path.basename(filename)}")
        data = pd.read_csv(filename)
        real_window = max(5, len(data) // 20)

        # 각 지표 플로팅
        score_ma = data['Score'].rolling(window=real_window).mean()
        ax[0, 0].plot(data['Episode'], score_ma, label=model, color=colors[model], linewidth=2)

        if 'Avg_Loss' in data.columns:
            loss_ma = data['Avg_Loss'].rolling(window=real_window).mean()
            ax[0, 1].plot(data['Episode'], loss_ma, label=model, color=colors[model], linewidth=2)

        wall_ma = data['Wall_Hits'].rolling(window=real_window).mean()
        ax[1, 0].plot(data['Episode'], wall_ma, label=model, color=colors[model], linewidth=2)

        coin_ma = data['Coins'].rolling(window=real_window).mean()
        ax[1, 1].plot(data['Episode'], coin_ma, label=model, color=colors[model], linewidth=2)

    if not found_any:
        print(f"❌ '{LOG_DIR}' 폴더에 비교할 로그 파일이 없습니다.")
        return

    # 그래프 세팅
    ax[0, 0].set_title('Score')
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    ax[0, 1].set_title('Loss')
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    ax[1, 0].set_title('Wall Hits')
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    ax[1, 1].set_title('Coins Eaten')
    ax[1, 1].legend()
    ax[1, 1].grid(True)

    plt.tight_layout()

    # [저장 기능] 비교 결과는 현재 시간으로 저장
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(LOG_DIR, f"comparison_result_{current_time}.png")
    plt.savefig(save_path)
    print(f"💾 비교 그래프 저장 완료: {save_path}")

    plt.show()

if __name__ == "__main__":
    if COMPARE_MODE:
        print(f"⚔️ 비교 모드 실행: '{LOG_DIR}' 폴더를 검색합니다...")
        plot_comparison()
    else:
        print(f"🔍 단일 모드 실행: {MODEL_TYPE} 분석 중...")
        plot_single_model(MODEL_TYPE)