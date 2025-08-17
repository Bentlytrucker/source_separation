import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import soundfile as sf

# --- 1. 설정값 (이전과 동일) ---
MIXTURE_FILE = "mix.wav"
FINGERPRINT_FILE = "learned_fingerprint.npy"
OUTPUT_FILE = "separated_target_sound.wav"
SAMPLE_RATE = 22050
N_FFT = 4096
HOP_LENGTH = 1024
N_MELS = 256
N_COMPONENTS_MIXTURE = 10

# --- 2. 핵심 함수 정의 (단순하고 강력한 방식으로 회귀) ---

def separate_target_sound(mixture_path, fingerprint_path, output_path, sr, n_fft, hop_length, n_mels, n_components):
    """'가장 유사한 단일 성분'을 타겟으로 특정하여 분리합니다."""
    
    print("1. 파일 로드 및 스펙트로그램 변환...")
    W_target_set = np.load(fingerprint_path)
    
    y_mix, _ = librosa.load(mixture_path, sr=sr)
    S_mix_orig_phase = librosa.stft(y=y_mix, n_fft=n_fft, hop_length=hop_length)
    S_mix_mag, _ = librosa.magphase(S_mix_orig_phase)

    print("2. 혼합 음원 NMF 분해...")
    model_mix = NMF(n_components=n_components, init='random', random_state=0, max_iter=500, solver='mu')
    W_mix = model_mix.fit_transform(S_mix_mag)
    H_mix = model_mix.components_

    print("3. 가장 유사한 '단일' 성분 탐색...")
    best_sim_score = -1.0
    best_match_index = -1

    # 혼합 음원의 각 성분(W_mix의 열)을 순회
    for i in range(W_mix.shape[1]):
        mix_component = W_mix[:, i]
        # 현재 성분과 모든 타겟 지문 간의 유사도를 계산하여 '최고점'을 찾음
        current_max_sim = 0
        for target_component in W_target_set:
            # 주파수 축 길이 맞추기
            target_component_resampled = np.interp(
                np.linspace(0, 1, len(mix_component)),
                np.linspace(0, 1, len(target_component)),
                target_component
            )
            sim = np.dot(target_component_resampled, mix_component) / (np.linalg.norm(target_component_resampled) * np.linalg.norm(mix_component))
            if sim > current_max_sim:
                current_max_sim = sim
        
        # 지금까지의 최고 유사도 점수보다 현재 성분의 점수가 더 높으면, 이 성분을 '가장 유력한 용의자'로 지목
        if current_max_sim > best_sim_score:
            best_sim_score = current_max_sim
            best_match_index = i

    print(f"   > 타겟 사운드는 혼합 음원의 {best_match_index + 1}번째 성분과 가장 유사합니다. (유사도: {best_sim_score:.4f})")

    print("4. 타겟 마스크 생성 및 음원 분리...")
    # '가장 유력한 용의자' 하나만 사용하여 타겟 스펙트로그램을 재구성
    W_target_reconstructed = W_mix[:, [best_match_index]]  # []를 사용해 2D 형태 유지
    H_target_reconstructed = H_mix[[best_match_index], :]
    S_target_reconstructed = W_target_reconstructed @ H_target_reconstructed
    
    # 소프트 마스크 생성
    mask = np.nan_to_num(S_target_reconstructed / (S_mix_mag + 1e-7))
    mask = np.minimum(mask, 1.0)
    
    # --- 핵심 수정: 타겟 마스크를 적용하여 타겟 사운드 분리 ---
    S_target_separated = S_mix_orig_phase * mask

    print("5. 분리된 음원 오디오로 복원...")
    y_target_separated = librosa.istft(S_target_separated, hop_length=hop_length)
    
    sf.write(output_path, y_target_separated, sr)
    print(f"\n## 분리 성공! ##\n분리된 파일이 '{output_path}'에 저장되었습니다.")
    
    # 시각화 (이전과 동일)
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

# --- 메인 실행부 (이전과 동일) ---
if __name__ == "__main__":
    separate_target_sound(MIXTURE_FILE, FINGERPRINT_FILE, OUTPUT_FILE, 
                          SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, N_COMPONENTS_MIXTURE)
