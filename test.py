import numpy as np
import scipy.io
import scipy.signal
import librosa
import matplotlib.pyplot as plt

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


data = scipy.io.loadmat('C:/Users/ovidiu/Downloads/fecgsyn2917.mat')
out_data = data['out']
mecg = out_data['mecg'][0, 0]
fecg = out_data['fecg'][0, 0]

ecg_signal = mecg.T + fecg.T
fecg=fecg.T
ecg_signal = ecg_signal.astype(float)
fecg=fecg.astype(float)

ecg_signal = scipy.signal.resample(ecg_signal, num=int(ecg_signal.shape[0] * (500 / 2500)), axis=0)
fecg = scipy.signal.resample(fecg, num=int(fecg.shape[0] * (500 / 2500)), axis=0)

random_start = 30 * 500
trim_sg = fecg[random_start:random_start+2000, 1]

n_fft = 128
hop_length = 48
win_length = 64
window = np.hanning(win_length)

spectrogram = librosa.stft(trim_sg, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
ref_spec = np.max(np.abs(spectrogram))
spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), ref=ref_spec)
# spectrogram = scale_minmax(spectrogram, 0, 255).astype(np.uint8)
# spectrogram = np.flip(spectrogram, axis=0)
# spectrogram = 255 - spectrogram


# spectrogram = 255 - spectsrogram
# spectrogram = np.flip(spectrogram, axis=0)
# spectrogram = scale_minmax(spectrogram, -80, 0).astype(np.float32)
# spectrogram *= ref_spec
spectrogram = librosa.db_to_amplitude(spectrogram, ref=ref_spec)
reconstructed_signal = librosa.istft(spectrogram, hop_length=hop_length, win_length=win_length, window=window)

print(f"Original Trimmed Signal Shape: {trim_sg.shape}")
print(f"Spectrogram Shape: {spectrogram.shape}")
print(f"Reconstructed Signal Shape: {reconstructed_signal.shape}")

trim_sg=trim_sg[:len(reconstructed_signal)]
print(np.allclose(reconstructed_signal, trim_sg))

plt.figure(), plt.plot(reconstructed_signal)
plt.plot(trim_sg, "r")


plt.show()