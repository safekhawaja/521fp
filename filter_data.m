function clean_data = filter_data_release(raw_eeg)
    %
    % filter_data_release.m
    %
    % Instructions: Write a filter function to clean underlying data.
    %               The filter type and parameters are up to you.
    %               Points will be awarded for reasonable filter type,
    %               parameters, and correct application. Please note there 
    %               are many acceptable answers, but make sure you aren't 
    %               throwing out crucial data or adversely distorting the 
    %               underlying data!
    %
    % Input:    raw_eeg (samples x channels)
    %
    % Output:   clean_data (samples x channels)
    % 
%% Your code here (2 points) 
%      load('filter.mat');
%      clean_data = filtfilt(coefs, 1, raw_eeg);
%     [~,d] = bandpass(raw_eeg, [0.15,200],1000);
% 
%     clean_data = filter(d,raw_eeg);
clean_data = raw_eeg;

end