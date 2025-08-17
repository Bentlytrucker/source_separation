import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

# --- 1. 설정값 (사용자 변경 필요) ---

# 20개의 순수 음원 wav 파일들이 들어있는 폴더 경로
# 예: "C:/data/my_sounds" 또는 "data/audio_samples"
# 하위 폴더가 아닌 해당 폴더에 wav 파일들이 바로 있어야 합니다.
AUDIO_DIR = "./data/Glass_Breaking" 

# 오디오 처리를 위한 파라미터 (일반적으로 이 값을 그대로 사용해도 좋습니다)
SAMPLE_RATE = 22050      # 샘플링 레이트
N_FFT = 4096            # FFT 윈도우 사이즈
HOP_LENGTH = 1024         # FFT 윈도우 간격
N_MELS = 256             # 멜 밴드 개수 (스펙트로그램의 세로 해상도)

# NMF를 통해 추출할 '음향 지문'의 개수
# 소리가 단일 요소를 갖는다면 1~5, 복합적인 소리라면 조금 더 높게 설정
N_COMPONENTS = 3

# 학습된 '음향 지문'을 저장할 파일 이름
FINGERPRINT_FILE = "learned_fingerprint.npy"


# --- 2. 1단계: 데이터 전처리 및 스펙트로그램 생성 ---

def create_spectrograms(audio_paths, sr, n_fft, hop_length, n_mels):
    """지정된 경로의 모든 오디오 파일을 스펙트로그램으로 변환합니다."""
    spectrograms = []
    print(f"총 {len(audio_paths)}개의 오디오 파일을 처리합니다...")
    
    for path in audio_paths:
        try:
            # 오디오 파일 로드
            y, _ = librosa.load(path, sr=sr)
            
            # 음량이 너무 작은 경우를 대비해 정규화
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            # 멜 스펙트로그램 계산
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            
            # 로그 스케일로 변환하여 dB 단위로 표현
            S_db = librosa.power_to_db(S, ref=np.max)
            spectrograms.append(S_db)
            
        except Exception as e:
            print(f"'{path}' 파일 처리 중 오류 발생: {e}")
            
    print("스펙트로그램 변환 완료.")
    return spectrograms


# --- 3. 2단계: '음향 지문' 학습 (NMF) ---

def learn_fingerprint(spectrograms, n_components, n_mels):
    """스펙트로그램들로부터 NMF를 이용해 음향 지문을 학습합니다."""
    # 모든 스펙트로그램을 시간 축(axis=1)을 따라 하나로 합치기
    full_spectrogram = np.concatenate(spectrograms, axis=1)
    
    # NMF는 음수 값을 처리할 수 없으므로, 모든 값을 0 이상으로 이동
    S_non_negative = full_spectrogram - np.min(full_spectrogram)

    print("\nNMF 모델 학습을 시작합니다 (데이터 크기에 따라 시간이 걸릴 수 있습니다)...")
    
    # NMF 모델 초기화
    # scikit-learn의 NMF는 (n_samples, n_features) 입력을 기대합니다.
    # 우리의 스펙트로그램은 (n_mels, n_frames) 이므로, n_samples=n_mels, n_features=n_frames 입니다.
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=500, solver='mu')
    
    # --- 핵심 수정 ---
    # fit_transform은 주파수 패턴 행렬(W)을 반환합니다. 이 행렬이 우리가 원하는 '음향 지문'입니다.
    # W 행렬의 형태: (n_samples, n_components) -> (n_mels, n_components) -> (128, 3)
    W = model.fit_transform(S_non_negative)
    
    # 시각화 및 후속 처리를 위해 (n_components, n_mels) 형태로 변환하여 반환합니다. -> (3, 128)
    fingerprint = W.T
    
    print("음향 지문 학습 완료.")
    return fingerprint


# --- 4. 시각화 및 저장 ---

# --- 4. 시각화 및 저장 (최종 시각화 수정 버전) ---

def visualize_and_save(fingerprint, sr, hop_length, n_mels):
    """학습된 음향 지문을 시각화하고 파일로 저장합니다. (로그 스케일 적용)"""
    # 파일 저장
    np.save(FINGERPRINT_FILE, fingerprint)
    print(f"\n학습된 음향 지문을 '{FINGERPRINT_FILE}' 파일로 저장했습니다.")

    # --- 데이터 검사 ---
    print("\n--- 시각화를 위한 데이터 검사 ---")
    print(f"Fingerprint 배열의 형태: {fingerprint.shape}")
    print(f"Fingerprint 전체 값 범위 (Min): {np.min(fingerprint):.6f}")
    print(f"Fingerprint 전체 값 범위 (Max): {np.max(fingerprint):.6f}")
    print("---------------------------------")
    
    # 시각화
    n_components = fingerprint.shape[0]
    plt.figure(figsize=(12, 4 * n_components))
    plt.suptitle("Learned Acoustic Fingerprints (Log Scale)", fontsize=16)

    for i in range(n_components):
        ax = plt.subplot(n_components, 1, i + 1)
        
        component = fingerprint[i, :]
        
        # --- 핵심 수정: 값을 dB(로그 스케일)로 변환하여 시각화 ---
        # power_to_db는 내부적으로 로그를 취해줘서 미세한 차이를 증폭시켜 보여줍니다.
        # ref=np.max를 통해 가장 밝은 부분을 0dB로 설정합니다.
        # component를 2D 배열로 만들어 함수에 전달합니다.
        component_db = librosa.power_to_db(component.reshape(n_mels, 1), ref=np.max)
        
        librosa.display.specshow(component_db, 
                                 sr=sr, 
                                 hop_length=hop_length, 
                                 x_axis='time', 
                                 y_axis='mel', 
                                 ax=ax)
        
        # 컬러바(범례)를 추가하여 dB 범위를 보여줍니다.
        fig = plt.gcf()
        fig.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
        
        ax.set_title(f"Component {i+1}")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("learned_fingerprint.png")
    print("음향 지문 그래프를 'learned_fingerprint.png' 파일로 저장했습니다.")
    plt.show()


# --- 메인 실행부 ---

if __name__ == "__main__":
    # 1. wav 파일 목록 가져오기
    if not os.path.isdir(AUDIO_DIR):
        print(f"오류: '{AUDIO_DIR}' 폴더를 찾을 수 없습니다. 폴더를 생성하고 wav 파일을 넣어주세요.")
    else:
        audio_files = glob.glob(os.path.join(AUDIO_DIR, '*.wav'))
        if not audio_files:
            print(f"오류: '{AUDIO_DIR}' 폴더에 wav 파일이 없습니다.")
        else:
            # 2. 1단계 실행
            all_spectrograms = create_spectrograms(audio_files, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS)
            
            if all_spectrograms:
                # 3. 2단계 실행 (N_MELS 파라미터 추가)
                learned_fingerprint = learn_fingerprint(all_spectrograms, N_COMPONENTS, N_MELS)
                
                # 4. 결과 저장 및 시각화
                visualize_and_save(learned_fingerprint, SAMPLE_RATE, HOP_LENGTH, N_MELS)
                
                print("\n## 모든 작업이 완료되었습니다. ##")
                print(f"이제 '{FINGERPRINT_FILE}' 파일을 사용해 3단계(음원 분리)를 진행할 수 있습니다.")
