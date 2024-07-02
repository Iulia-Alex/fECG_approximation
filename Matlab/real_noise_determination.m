x1 = removeSaturation(sampledata_input1, ADC1sat);
x2 = removeSaturation(sampledata_input2, ADC2sat);
x3 = removeSaturation(sampledata_input3, ADC3sat);
x4 = removeSaturation(sampledata_input4, ADC4sat);
y1 = x1 - x4; y2 = x2 - x4 ; y3 = x3 - x4; y4 = x1 - x3;
figure; subplot(2,2,1); plot(y1); title('Initial y1'); xlabel('Samples');
subplot(2,2,2); plot(y3); title('Initial y3'); xlabel('Samples');
signal=[removeArtefacts(y1), removeArtefacts(y2), removeArtefacts(y3), removeArtefacts(y4)]; %signal = y3;
subplot(2,2,3); plot(signal(:,1)); title('y1 after Artefact Removal'); xlabel('Samples'); 
subplot(2,2,4); plot(signal(:,3)); title('y3 after Artefact Removal'); xlabel('Samples');

sr = 2500;
% Parametri pentru filtrul notch
centerFreqNotch = 50; % Frecvența centrală a notch filter
bwNotch = 1; % Lățimea de bandă a notch filter

% Creare filtru notch
notchFilter = designfilt('bandstopiir', 'FilterOrder', 10, 'HalfPowerFrequency1', centerFreqNotch - bwNotch/2, 'HalfPowerFrequency2', centerFreqNotch + bwNotch/2, 'SampleRate', sr);

% Parametri pentru filtrul bandpass
lowFreqBP = 1; % Frecvența minimă
highFreqBP = 100; % Frecvența maximă

% Creare filtru bandpass
bandpassFilter = designfilt('bandpassiir', 'FilterOrder', 10, 'HalfPowerFrequency1', lowFreqBP, 'HalfPowerFrequency2', highFreqBP, 'SampleRate', sr);

% Aplicare filtru notch
filteredSignalNotch = filtfilt(notchFilter, signal);

% Aplicare filtru bandpass
filteredSignalBandpass = filtfilt(bandpassFilter, filteredSignalNotch);

% figure; plot(signal); title('ECG initial signals without artefacts');

% Răspunsul în frecvență pentru filtrul notch
freqNotch = freqz(notchFilter);

% Răspunsul în frecvență pentru filtrul bandpass
freqBandpass = freqz(bandpassFilter);

% figure;
% subplot(2,1,1);
% plot(linspace(0, sr/2, length(freqNotch)), 20*log10(abs(freqNotch)));
% title('Frequency response - Notch Filter');
% xlabel('Freq (Hz)');
% ylabel('Amplitude (dB)');
% grid on;
% 
% subplot(2,1,2);
% plot(linspace(0, sr/2, length(freqBandpass)), 20*log10(abs(freqBandpass)));
% title('Frequency response - Bandpass Filter');
% xlabel('Freq (Hz)');
% ylabel('Amplitude (dB)');
% grid on;

t = 1:150000; % Intervalul de timp

% Plotare semnal original și semnal filtrat
figure;tiledlayout(3,1);
ax1 = nexttile;
% subplot(3,1,1);
plot(t, signal(:,1)); xlabel('Samples');
title('Original signal - Channel 1');

% subplot(3,1,2);
ax2 = nexttile;
plot(t, filteredSignalNotch(:,1)); xlabel('Samples');
title('Notch-filtered signal - Channel 1');

% subplot(3,1,3);
ax3 = nexttile;
plot(t, filteredSignalBandpass(:,1)); xlabel('Samples');
title('Bandpass-filtered signal - Channel 1');
linkaxes([ax1 ax2 ax3],'x')

% Raspunsul in frecventa al semnalului filtrat
N_filt = length(filteredSignalBandpass); % Lungimea semnalului
frequencies_filt = linspace(0, sr/2, N_filt/2+1); % Vector de frecvențe corespunzător FFT
fft_result_filt = fft(filteredSignalBandpass) / N_filt; % Calculul FFT și normalizare

figure; plot(frequencies_filt, 2*abs(fft_result_filt(1:N_filt/2+1, 1))); % Afisare raspuns in frecventa
title('Frequency spectrum of notch + bandpass filtered signal, channel 1'); xlabel('Frequency [Hz]');

fft_result_notch = fft(filteredSignalNotch);
figure; plot(frequencies_filt, 2*abs(fft_result_notch(1:N_filt/2+1, 1)/N_filt)); title('Freq spectrum of notched signal, channel 1');

independent_components = ecgICA(filteredSignalBandpass, sr);

figure; subplot(2,2,1); plot(independent_components(:,1)); title('Component 1'); xlabel('Samples');
subplot(2,2,2); plot(independent_components(:,2)); title('Component 2'); xlabel('Samples');
subplot(2,2,3); plot(independent_components(:,3)); title('Component 3'); xlabel('Samples');
subplot(2,2,4); plot(independent_components(:,4)); title('Component 4'); xlabel('Samples'); sgtitle('ICA results');

% peaks = OSET_MaxSearch(filteredSignalBandpass,1/sr, 1); % 80/sr?
% [qrs_pos,sign,en_thres] = jqrs_mod(independent_components)
peaks1 = pantompkins_qrs(independent_components(:,1), sr);
peaks2 = pantompkins_qrs(independent_components(:,2), sr);
peaks3 = pantompkins_qrs(independent_components(:,3), sr);
peaks4 = pantompkins_qrs(independent_components(:,4), sr);

figure;
sgtitle('Identified peaks on each component');
subplot(2,2,1); plot(peaks1, independent_components(peaks1, 1), 'ro', 'MarkerSize', 8);
hold on; plot(independent_components(:, 1)); title('Component 1'); xlabel('Samples');
subplot(2,2,2); plot(peaks2, independent_components(peaks2, 2), 'ro', 'MarkerSize', 8);
hold on; plot(independent_components(:, 2)); title('Component 2'); xlabel('Samples');
subplot(2,2,3); plot(peaks3, independent_components(peaks3, 1), 'ro', 'MarkerSize', 8);
hold on; plot(independent_components(:, 3)); title('Component 3'); xlabel('Samples');
subplot(2,2,4); plot(peaks4, independent_components(peaks4, 1), 'ro', 'MarkerSize', 8);
hold on; plot(independent_components(:, 4)); title('Component 4'); xlabel('Samples');

nr_channels = 4;
peaks = {peaks1, peaks2, peaks3, peaks4};
[Qmin, channel] = Qindex(nr_channels, peaks);
%noise = ECGremoval(filteredSignalNotch, peaks{channel},sr,'',0,0,0,3,0);
noise = ECGremoval(filteredSignalNotch, peaks{channel},sr,'',0,0);

noise = noise(600:end-600, :); t = linspace(0, length(noise), length(noise));

figure; sgtitle('Noise obtained from bipolar signals')
for i = 1:4
    subplot(2, 2, i); plot(t, noise(:, i)); xlabel('Samples');
    xlabel('Samples');
end

figure; subplot(2,1,1); plot(independent_components(:, channel)); title('Selected channel'); xlabel('Samples');
subplot(2,1,2); plot(noise(:,channel)); xlabel('Samples');

noise_freq = abs(fft(noise));
figure; sgtitle('Frequency spectrum of the noise obtained from bipolar signals');
for i = 1:4
    subplot(2, 2, i); plot(frequencies_filt, 2*abs(noise_freq(1:N_filt/2+1, i))); % Afisare raspuns in frecventa;
    xlabel('Frequency [Hz]');
end

% The power in each frequency band
% nr_bands = 10; % numărul de benzi de frecvență
% bands = linspace(0, sr/2, nr_bands+1); % limitele benzilor de frecvență
% 
% bandpower = zeros(4, nr_bands);
% for j = 1:4
%     for i = 1:nr_bands
%         begin = round(bands(i) * length(noise_freq) / sr) + 1;
%         endd = round(bands(i+1) * length(noise_freq) / sr);
%         bandpower(i) = sum(abs(noise_freq(begin:endd, j)).^2);
%     end
% end
% 
% bpf = designfilt('bandpassiir', 'FilterOrder', 18, 'HalfPowerFrequency1', 1, 'HalfPowerFrequency2', 10, 'SampleRate', sr);
% Pink = filtfilt(bpf, noise);
% figure; sgtitle('Pink Noise obtained - in Time Domain')
% for i = 1:4
%     subplot(2, 2, i); plot(t, Pink(:, i));
% end
% Pink_freq = abs(fft(Pink));
% figure; sgtitle('Frequency spectrum of the Pink noise obtained');
% for i = 1:4
%     subplot(2, 2, i); plot(frequencies_filt, 2*abs(Pink_freq(1:N_filt/2+1, i))); % Afisare raspuns in frecventa;
% end
% correlation_Pink = corr(Pink);
% 
% bpf = designfilt('bandpassiir', 'FilterOrder', 18, 'HalfPowerFrequency1', 10, 'HalfPowerFrequency2', 70, 'SampleRate', sr);
% White = filtfilt(bpf, noise);
% figure; sgtitle('White Noise obtained - in Time Domain')
% for i = 1:4
%     subplot(2, 2, i); plot(t, White(:, i));
% end
% White_freq = abs(fft(White));
% figure; sgtitle('Frequency spectrum of the White noise obtained');
% for i = 1:4
%     subplot(2, 2, i); plot(frequencies_filt, 2*abs(White_freq(1:N_filt/2+1, i))); % Afisare raspuns in frecventa;
% end
% correlation_White = corr(White);
% 
% bpf = designfilt('bandpassiir', 'FilterOrder', 18, 'HalfPowerFrequency1', 70, 'HalfPowerFrequency2', 120, 'SampleRate', sr);
% Mixt = filtfilt(bpf, noise);
% figure; sgtitle('Mixt Noise obtained - in Time Domain')
% for i = 1:4
%     subplot(2, 2, i); plot(t, Mixt(:, i));
% end
% Mixt_freq = abs(fft(Mixt));
% figure; sgtitle('Frequency spectrum of the Mixt noise obtained');
% for i = 1:4
%     subplot(2, 2, i); plot(frequencies_filt, 2*abs(Mixt_freq(1:N_filt/2+1, i))); % Afisare raspuns in frecventa;
% end
% correlation_Mixt = corr(Mixt);
% 
% bpf = designfilt('bandpassiir', 'FilterOrder', 18, 'HalfPowerFrequency1', 120, 'HalfPowerFrequency2', 1000, 'SampleRate', sr);
% interf = filtfilt(bpf, noise);
% figure; sgtitle('interf Noise obtained - in Time Domain')
% for i = 1:4
%     subplot(2, 2, i); plot(t, interf(:, i));
% end
% interf_freq = abs(fft(interf));
% figure; sgtitle('Frequency spectrum of the interf noise obtained');
% for i = 1:4
%     subplot(2, 2, i); plot(frequencies_filt, 2*abs(interf_freq(1:N_filt/2+1, i))); % Afisare raspuns in frecventa;
% end
% correlation_interf = corr(interf);
% correlation_interf_freq = corr(interf_freq);
% 
signal1=[removeArtefacts(x1), removeArtefacts(x2), removeArtefacts(x3), removeArtefacts(x4)]; 
filteredSignalNotch1 = filtfilt(notchFilter, signal1);
noise1 = ECGremoval(filteredSignalNotch1, peaks{channel},sr,'',0,0);
noise1 = noise1(600:end-600, :);
noise_freq1 = abs(fft(noise1));

figure; sgtitle('Noise obtained from unipolar signals')
for i = 1:4
    subplot(2, 2, i); plot(t, noise1(:, i));
    xlabel('Samples');
end

figure; sgtitle('Frequency spectrum of the noise obtained from unipolar signals');
for i = 1:4
    subplot(2, 2, i); plot(frequencies_filt, 2*abs(noise_freq1(1:N_filt/2+1, i))); % Afisare raspuns in frecventa;
    xlabel('Frequency [Hz]');
end
% 
% bpf = designfilt('bandpassiir', 'FilterOrder', 18, 'HalfPowerFrequency1', 1, 'HalfPowerFrequency2', 10, 'SampleRate', sr);
% Pink1 = filtfilt(bpf, noise1);
% figure; sgtitle('Pink Noise obtained - in Time Domain')
% for i = 1:4
%     subplot(2, 2, i); plot(t, Pink1(:, i));
% end
% Pink_freq1 = abs(fft(Pink1));
% figure; sgtitle('Frequency spectrum of the Pink noise obtained');
% for i = 1:4
%     subplot(2, 2, i); plot(frequencies_filt, 2*abs(Pink_freq1(1:N_filt/2+1, i))); % Afisare raspuns in frecventa;
% end
% correlation_Pink1 = corr(Pink1);
% 
% bpf = designfilt('bandpassiir', 'FilterOrder', 18, 'HalfPowerFrequency1', 10, 'HalfPowerFrequency2', 70, 'SampleRate', sr);
% White1 = filtfilt(bpf, noise1);
% figure; sgtitle('White Noise obtained - in Time Domain')
% for i = 1:4
%     subplot(2, 2, i); plot(t, White1(:, i));
% end
% White_freq1 = abs(fft(White1));
% figure; sgtitle('Frequency spectrum of the White noise obtained');
% for i = 1:4
%     subplot(2, 2, i); plot(frequencies_filt, 2*abs(White_freq1(1:N_filt/2+1, i))); % Afisare raspuns in frecventa;
% end
% correlation_White1 = corr(White1);
% 
% bpf = designfilt('bandpassiir', 'FilterOrder', 18, 'HalfPowerFrequency1', 70, 'HalfPowerFrequency2', 120, 'SampleRate', sr);
% Mixt1 = filtfilt(bpf, noise1);
% figure; sgtitle('Mixt Noise obtained - in Time Domain')
% for i = 1:4
%     subplot(2, 2, i); plot(t, Mixt1(:, i));
% end
% Mixt_freq1 = abs(fft(Mixt1));
% figure; sgtitle('Frequency spectrum of the Mixt noise obtained');
% for i = 1:4
%     subplot(2, 2, i); plot(frequencies_filt, 2*abs(Mixt_freq1(1:N_filt/2+1, i))); % Afisare raspuns in frecventa;
% end
% correlation_Mixt1 = corr(Mixt1);
% 
% bpf = designfilt('bandpassiir', 'FilterOrder', 18, 'HalfPowerFrequency1', 120, 'HalfPowerFrequency2', 1000, 'SampleRate', sr);
% interf1 = filtfilt(bpf, noise1);
% figure; sgtitle('interf Noise obtained - in Time Domain')
% for i = 1:4
%     subplot(2, 2, i); plot(t, interf1(:, i));
% end
% interf_freq1 = abs(fft(interf1));
% figure; sgtitle('Frequency spectrum of the interf noise obtained');
% for i = 1:4
%     subplot(2, 2, i); plot(frequencies_filt, 2*abs(interf_freq1(1:N_filt/2+1, i))); % Afisare raspuns in frecventa;
% end
% correlation_interf1 = corr(interf1);

% %noiseBP = ECGremoval(filteredSignalBandpass,
% %peaks{channel},sr,'',0,0,0,3,0);
% noiseBP = ECGremoval(filteredSignalBandpass, peaks{channel},sr,'',0,0);
% noiseBP = noiseBP(600:end-600, :);
% 
% figure; sgtitle('Noise (trimmed) obtained - ECGremoval on Bandpassed signal')
% for i = 1:4
%     subplot(2, 2, i); plot(t, noiseBP(:, i));
% end
% 
% figure; subplot(2,1,1); plot(independent_components(:, channel)); title('Selected channel - noisebp');
% subplot(2,1,2); plot(noiseBP(:,channel));
% 
% noiseBP_freq = abs(fft(noiseBP));
% figure; sgtitle('Frequency response of the noise obtained from Bandpassed signal');
% for i = 1:4
%     subplot(2, 2, i); plot(frequencies_filt, 2*abs(noiseBP_freq(1:N_filt/2+1, i))); % Afisare raspuns in frecventa;
% end
% 
% % The power in each frequency band
% bandsBP = linspace(0, sr/2, nr_bands+1); % limitele benzilor de frecvență
% 
% bandpowerBP = zeros(4, nr_bands);
% for j = 1:4
%     for i = 1:nr_bands
%         begin = round(bandsBP(i) * length(noiseBP_freq) / sr) + 1;
%         endd = round(bandsBP(i+1) * length(noiseBP_freq) / sr);
%         bandpowerBP(i) = sum(abs(noiseBP_freq(begin:endd, j)).^2);
%     end
% end
% 
% figure; subplot(2,2,3); plot(noiseBP(:, channel)); title('Noise bp best channel obtained - nds = 3');
% noiseBP = ECGremoval(filteredSignalBandpass, peaks{channel},sr,'',0,0,0,1,0);
% subplot(2,2,1); plot(noiseBP(:, channel)); title('Noise bp best channel obtained - nds = 1');
% noiseBP = ECGremoval(filteredSignalBandpass, peaks{channel},sr,'',0,0,0,2,0);
% subplot(2,2,2); plot(noiseBP(:, channel)); title('Noise bp best channel obtained - nds = 2');
% noiseBP = ECGremoval(filteredSignalBandpass, peaks{channel},sr,'',0,0,0,5,0);
% subplot(2,2,4); plot(noiseBP(:, channel)); title('Noise bp best channel obtained - nds = 5');
% disp('Best channel is'); disp(channel);


% S = stft(signal(:, 2), 500, 'Window', hann(2000), 'OverlapLength', 1968, 'FFTLength', 2024);
% [X,T] = istft(S,500,'Window',hann(2000),'OverlapLength',1968,'FFTLength',2024,'Method','ola');
% figure; plot(T, X);
% overlap length nu e hop length, ovl=winl-hopl
% dim stft e nfftx(sgn_size - win_length)/hop_length