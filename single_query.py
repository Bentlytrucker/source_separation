import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import soundfile as sf

# --- 1. 설정값 (사용자 변경 필요) ---
# '음향 지문'으로 사용할 단 하나의 순수 타겟 음원 파일
TARGET_SAMPLE_FILE = "/data/Glass_Breaking/glass_1.wav"
# 분리할 대상인, 여러 소리가 섞인 혼합 음원 파일
MIXTURE_FILE = "mix.wav"
# 분리된 타겟 사운드를 저장할 파일 이름
OUTPUT_FILE = "separated_single_sample.wav"

# 오디오 처리를 위한 파라미터 (고해상도 버전)
SAMPLE_RATE = 22050
N_FFT = 4096
HOP_LENGTH = 1024
N_COMPONENTS_MIXTURE = 10 # 혼합 음원에서 분해할 전체 성분 개수

# --- 2. 핵심 함수 정의 ---

def create_fingerprint_from_single_file(file_path, sr, n_fft, hop_length):
    """단일 오디오 파일로부터 평균적인 주파수 패턴(음향 지문)을 생성합니다."""
    try:
        y, _ = librosa.load(file_path, sr=sr)
        S_mag, _ = librosa.magphase(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        # 스펙트로그램의 시간 축에 대한 평균을 계산하여 대표 주파수 패턴을 만듦
        fingerprint = np.mean(S_mag, axis=1)
        return fingerprint
    except FileNotFoundError:
        print(f"오류: 타겟 샘플 파일 '{file_path}'를 찾을 수 없습니다.")
        return None

def separate_with_single_sample(mixture_path, target_sample_path, output_path, sr, n_fft, hop_length, n_components):
    """단일 샘플 음원을 지문으로 사용하여 혼합 음원에서 타겟 소리를 분리합니다."""
    
    print("1. 단일 샘플로부터 음향 지문 생성...")
    w_target_fingerprint = create_fingerprint_from_single_file(target_sample_path, sr, n_fft, hop_length)
    if w_target_fingerprint is None:
        return

    print("2. 혼합 음원 로드 및 NMF 분해...")
    y_mix, _ = librosa.load(mixture_path, sr=sr)
    S_mix_orig_phase = librosa.stft(y=y_mix, n_fft=n_fft, hop_length=hop_length)
    S_mix_mag, _ = librosa.magphase(S_mix_orig_phase)
    
    model_mix = NMF(n_components=n_components, init='random', random_state=0, max_iter=500, solver='mu')
    W_mix = model_mix.fit_transform(S_mix_mag)
    H_mix = model_mix.components_

    print("3. 음향 지문과 가장 유사한 단일 성분 탐색...")
    best_sim_score = -1.0
    best_match_index = -1
    for i in range(W_mix.shape[1]):
        mix_component = W_mix[:, i]
        sim = np.dot(w_target_fingerprint, mix_component) / (np.linalg.norm(w_target_fingerprint) * np.linalg.norm(mix_component))
        if sim > best_sim_score:
            best_sim_score = sim
            best_match_index = i
    
    print(f"   > 타겟 사운드는 혼합 음원의 {best_match_index + 1}번째 성분과 가장 유사합니다. (유사도: {best_sim_score:.4f})")

    print("4. 마스크 생성 및 음원 분리...")
    W_target_reconstructed = W_mix[:, [best_match_index]]
    H_target_reconstructed = H_mix[[best_match_index], :]
    S_target_reconstructed = W_target_reconstructed @ H_target_reconstructed
    
    mask = np.nan_to_num(S_target_reconstructed / (S_mix_mag + 1e-7))
    mask = np.minimum(mask, 1.0)
    
    S_target_separated = S_mix_orig_phase * mask

    print("5. 분리된 음원 오디오로 복원...")
    y_target_separated = librosa.istft(S_target_separated, hop_length=hop_length)
    sf.write(output_path, y_target_separated, sr)
    print(f"\n## 분리 성공! ##\n분리된 파일이 '{output_path}'에 저장되었습니다.")
    
    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(10, 8))
    librosa.display.specshow(librosa.amplitude_to_db(S_mix_mag, ref=np.max), y_axis='log', x_axis='time', ax=ax[0], sr=sr, hop_length=hop_length)
    ax[0].set(title='Original Mixture')
    separated_mag, _ = librosa.magphase(S_target_separated)
    librosa.display.specshow(librosa.amplitude_to_db(separated_mag, ref=np.max), y_axis='log', x_axis='time', ax=ax[1], sr=sr, hop_length=hop_length)
    ax[1].set(title='Separated Target Sound')
    S_background = S_mix_orig_phase * (1 - mask)
    background_mag, _ = librosa.magphase(S_background)
    librosa.display.specshow(librosa.amplitude_to_db(background_mag, ref=np.max), y_axis='log', x_axis='time', ax=ax[2], sr=sr, hop_length=hop_length)
    ax[2].set(title='Separated Background')
    plt.tight_layout()
    plt.savefig("separation_result.png")
    plt.show()

# --- 메인 실행부 ---
if __name__ == "__main__":
    separate_with_single_sample(MIXTURE_FILE, TARGET_SAMPLE_FILE, OUTPUT_FILE, 
                                SAMPLE_RATE, N_FFT, HOP_LENGTH, N_COMPONENTS_MIXTURE)
