function signal = removeArtefacts(signal)

differences = abs(diff(signal));
i = 1;
threshold = 0.001;

while i < length(differences)
    if differences(i) > threshold
        delta = abs(differences(i+1) - differences(i));
        % disp(i);
        for j = i+2:length(signal)
            signal(j) = signal(j) - delta;
        end
        
        signal(i + 1) = (signal(i) + signal(i+2)) / 2;
        i = i + 2;
    else
        i = i + 1;
    end
end
end

% for i = 3:length(signal)-2
%     if (signal(i) < (sum(signal(i - 2:i + 2)) - signal(i))/4)
%         signal(i) = (sum(signal(i - 2:i + 2)) - signal(i))/4;
%     end
% end

% differences = abs(diff(signal));
% for i = 1: length(differences)
%     if differences(i) > 0.001
%         delta = abs(differences(i+1) - differences(i))
%         disp(i)
%         for j=i+2:length(signal)
%             signal(j) = signal(j) - delta;
%         end
%         i = i + 2
%     end