function [Qmin, channel] = Qindex(nr_channels, peaks)
    Q = inf(nr_channels, 1);
    
    selected_peaks = {};

    if ~isempty(peaks)
        for i = 1: nr_channels
            peaks{i} = peaks{i} / 2500;
            hr_segment = 60 ./ diff(peaks{i});
            mhr_mean = mean(hr_segment);
            
            if mhr_mean >= 50 && mhr_mean <= 180
                outliers = numel(hr_segment(hr_segment < 50 | hr_segment > 180)) / numel(peaks{i});
                variability = sum(abs(diff(hr_segment))) / sum(hr_segment);
                discrepancy = sum((hr_segment - mhr_mean) / numel(peaks{i})) / mhr_mean;
                Q(i) = outliers + variability + discrepancy;
                
                selected_peaks{end+1} = peaks{i};
            end
        end
    end
    
    if isempty(selected_peaks)
        disp('Toate seriile au fost eliminate. mECG removal nu va fi efectuat.');
        Qmin = NaN;
        channel = NaN;
    else
        [Qmin, channel] = min(Q);
    end
end