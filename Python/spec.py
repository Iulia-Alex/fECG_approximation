import os
import librosa
import numpy as np
import scipy.io as sio
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import colorednoise as cn

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def mixtgauss(N, p, sigma0, sigma1):
    ''' 
    gives a mixture of gaussian noise
    arguments:
    N: data length
    p: probability of peaks
    sigma0: standard deviation of background noise
    sigma1: standard deviation of impulse noise
    output: x: output noise
    '''
    q = np.random.randn(N)
    u = q < p
    x = (sigma0 * (1 - u) + sigma1 * u) * np.random.randn(N)
    return x

def add_noise(ecg, num_channels, sample_rate, nb_samples):
    ecg_plus_noise = ecg
    noise_arr = np.empty(ecg.shape)
    for i in range(num_channels):
        beta = 1  
        pink_noise = cn.powerlaw_psd_gaussian(beta, nb_samples)
        white_noise = np.random.randn(nb_samples)
        mixture_gaussian = mixtgauss(nb_samples, 0.1, 1, 10)

        pink_fft = np.fft.fft(pink_noise)
        white_fft = np.fft.fft(white_noise)
        mixture_fft = np.fft.fft(mixture_gaussian)

        pink_range = int(np.random.rand() * 4 + 9)
        num_samples_pink = pink_range * nb_samples // sample_rate
        white_range = int(np.random.rand() * 30 + 60)
        num_samples_white = white_range * nb_samples // sample_rate

        pink_fft =  pink_fft / max(np.abs(pink_fft[:num_samples_pink]))
        white_fft = white_fft / max(np.abs(white_fft[num_samples_pink:num_samples_pink+num_samples_white]))
        mixture_fft = mixture_fft / max(np.abs(mixture_fft[num_samples_pink+num_samples_white:]))
        combined_fft = np.zeros(nb_samples, dtype=np.complex128)
        combined_fft += 2*pink_fft + 0.2*white_fft + 0.15*mixture_fft
        noise_timedomain = np.fft.ifft(combined_fft)
        
        snr_db = np.random.rand() * 10 + 50
        rms_signal = np.sqrt(np.mean(np.square(ecg)))
        rms_noise_exp = 10 ** (np.log10(rms_signal) - snr_db / 20)
        normalized_noise = (noise_timedomain - np.mean(noise_timedomain)) / np.std(noise_timedomain)
        new_noise = rms_noise_exp * normalized_noise
        ecg_plus_noise[:, i] += new_noise.astype(float)
        noise_arr[:, i] = new_noise
    return (ecg_plus_noise, noise_arr)

data = sio.loadmat("/home/iulia/Desktop/iulia/Licenta/unet/ecg/fecgsyn01.mat")
out_data = data['out']
mecg = out_data['mecg'][0, 0]
fecg = out_data['fecg'][0, 0]
ecg_signal = mecg.T + fecg.T
ecg_signal = ecg_signal.astype(float)
# plt.plot(ecg_signal[:, 1])
ecg_signal = librosa.resample(y=ecg_signal, orig_sr=2500, target_sr=500, axis=0)
plt.figure()
plt.plot(ecg_signal[:, 1])
(ecg_plus_noise, noise) = add_noise(ecg_signal, 4, 500, 30000)
# plt.figure()
# plt.plot(ecg_plus_noise[:, 1])
# plt.figure()
# plt.plot(noise[:, 1])

i=1
spectrogram = librosa.stft(ecg_plus_noise[:, i], win_length=128, n_fft=128, hop_length=32)
spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
spectrogram = scale_minmax(spectrogram, 0, 255).astype(np.uint8)
spectrogram = np.flip(spectrogram, axis=0) 
spectrogram = 255 - spectrogram
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
img = librosa.display.specshow(spectrogram, y_axis='linear', x_axis='time', sr=500, ax=ax[0])
ax[0].set(title='Linear-frequency power spectrogram')
ax[0].label_outer()

hop_length = 32
librosa.display.specshow(spectrogram, y_axis='log', sr=500, hop_length=hop_length, x_axis='time', ax=ax[1])
ax[1].set(title='Log-frequency power spectrogram')
ax[1].label_outer()
fig.colorbar(img, ax=ax, format="%+2.f dB")
