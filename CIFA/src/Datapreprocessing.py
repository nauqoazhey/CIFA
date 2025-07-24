import librosa
import librosa.display
import os
import numpy as np
from math import sqrt
from scipy.signal import butter, lfilter


def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, sr = librosa.load(file_path, sr=None)
    return x, sr

def normalization(x):
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean) / std
    return x

def calculate_melsp(audio, sr, enhance_high=False):
    target_frames = 64
    hop_length = int(len(audio) / (target_frames - 1))
    melspec = librosa.feature.melspectrogram(y=audio, sr=sr, S=None, n_fft=1024, hop_length=hop_length, n_mels=128, window='hann', center=True, pad_mode='reflect', power=2.0)
    logmelspec = librosa.power_to_db(melspec)
    return logmelspec


def min_max_normal(data):
    min_value = np.min(data)
    max_value = np.max(data)
    if max_value - min_value == 0:
        return np.zeros_like(data)
    return (data - min_value) / (max_value - min_value)


def HP_melsp(y, sr, enhance_high=False):
    target_frames = 64
    hop_length = int(len(y) / (target_frames - 1))
    D = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=hop_length, n_mels=128, window='hann', center=True, pad_mode='reflect', power=2.0)
    log_D = librosa.amplitude_to_db(D, ref=np.max)
    H, P = librosa.decompose.hpss(np.abs(log_D))
    return H, P


def HP_melsp2(y, sr):
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    h_melsp = calculate_melsp(y_harmonic, sr)
    p_melsp = calculate_melsp(y_percussive, sr)

    return h_melsp, p_melsp


def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    lam = np.random.uniform(0, alpha)
    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12