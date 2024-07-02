function out = real_noise_determination_function(ADC1sat, ADC2sat, ADC3sat, ADC4sat, sampledata_input1, sampledata_input2, sampledata_input3, sampledata_input4)
    x1 = removeSaturation(sampledata_input1, ADC1sat);
    x2 = removeSaturation(sampledata_input2, ADC2sat);
    x3 = removeSaturation(sampledata_input3, ADC3sat);
    x4 = removeSaturation(sampledata_input4, ADC4sat);
    y1 = x1 - x4; y2 = x2 - x4 ; y3 = x3 - x4; y4 = x1 - x3;

    signal=[removeArtefacts(y1), removeArtefacts(y2), removeArtefacts(y3), removeArtefacts(y4)]; %signal = y3;

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

    independent_components = ecgICA(filteredSignalBandpass, sr);

    % peaks = OSET_MaxSearch(filteredSignalBandpass,1/sr, 1); % 80/sr?
    % [qrs_pos,sign,en_thres] = jqrs_mod(independent_components)
    peaks1 = pantompkins_qrs(independent_components(:,1), sr);
    peaks2 = pantompkins_qrs(independent_components(:,2), sr);
    peaks3 = pantompkins_qrs(independent_components(:,3), sr);
    peaks4 = pantompkins_qrs(independent_components(:,4), sr);

    nr_channels = 4;
    peaks = {peaks1, peaks2, peaks3, peaks4};
    [Qmin, channel] = Qindex(nr_channels, peaks);
    noise = ECGremoval(filteredSignalNotch, peaks{channel},sr,'',0,0);
    noise = noise(600:end-600, :); t = linspace(0, length(noise), length(noise));
    noise_freq = abs(fft(noise));

    % The power in each frequency band
    nr_bands = 10; % numărul de benzi de frecvență
    bands = linspace(0, sr/2, nr_bands+1); % limitele benzilor de frecvență

    bandpower = zeros(4, nr_bands);
    for j = 1:4
        for i = 1:nr_bands
            begin = round(bands(i) * length(noise_freq) / sr) + 1;
            endd = round(bands(i+1) * length(noise_freq) / sr);
            bandpower(i) = sum(abs(noise_freq(begin:endd, j)).^2);
        end
    end
    
    figure; sgtitle('Noise obtained - ECGremoval on Notched signal')
    for i = 1:4
        subplot(2, 2, i); plot(t, noise(:, i));
    end

    noise_freq = abs(fft(noise));
    N_filt = length(filteredSignalBandpass); % Lungimea semnalului
    frequencies_filt = linspace(0, sr/2, N_filt/2+1); % Vector de frecvențe corespunzător FFT
    figure; sgtitle('Frequency spectrum of the noise obtained from Notched signal');
    for i = 1:4
        subplot(2, 2, i); plot(frequencies_filt, 2*abs(noise_freq(1:N_filt/2+1, i))); % Afisare raspuns in frecventa;
    end
    
    disp('Best channel is'); disp(channel);
    out = {noise, channel};
    
    director_nou = 'noise_obtained';
    cale_completa = fullfile(pwd, director_nou);
    if ~exist(cale_completa, 'dir')
        mkdir(cale_completa);
    end
    
    save(fullfile(cale_completa, sprintf('noise%2.2d%2.2ddB_l%d_c0.mat', i)), 'out');
    nume_random = randi([100, 999]);
    nume_fisier = fullfile(cale_completa, sprintf('noise%d.mat', nume_random));

    if ~exist(nume_fisier, 'file')
        save(nume_fisier, 'out');
    else
        while exist(nume_fisier, 'file')
            nume_random = randi([100, 999]);
            nume_fisier = fullfile(cale_completa, sprintf('noise%d.mat', nume_random));
        end
        save(nume_fisier, 'out');
    end
end