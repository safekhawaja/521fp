function [all_feats]=getWindowedFeats_release(raw_data, fs, window_length, window_overlap)
    %
    % getWindowedFeats_release.m
    %
    % Instructions: Write a function which processes data through the steps
    %               of filtering, feature calculation, creation of R matrix
    %               and returns features.
    %
    %               Points will be awarded for completing each step
    %               appropriately (note that if one of the functions you call
    %               within this script returns a bad output you won't be double
    %               penalized)
    %
    %               Note that you will need to run the filter_data and
    %               get_features functions within this script. We also 
    %               recommend applying the create_R_matrix function here
    %               too.
    %
    % Inputs:   raw_data:       The raw data for all patients
    %           fs:             The raw sampling frequency
    %           window_length:  The length of window
    %           window_overlap: The overlap in window
    %
    % Output:   all_feats:      All calculated features
    %
%% Your code here (3 points)

% First, filter the raw data
cleaned_data = filter_data(raw_data);

avg = mean(cleaned_data');
for i = 1:size(cleaned_data, 1)
    for j = 1:size(cleaned_data, 2)
        cleaned_data(i,j) = cleaned_data(i,j) - avg;
    end
end

% Then, loop through sliding windows
NumWins = @(xLen, fs, winLen, winDisp) round((xLen/fs-winLen)/winDisp+1);
windows = NumWins(length(cleaned_data), fs, window_length, window_overlap);

    % Within loop calculate feature for each segment (call get_features)

% Finally, return feature matrix

tmp = get_features(cleaned_data, fs);
all_feats = zeros(windows, length(tmp));
start = 1;

for i = 1:windows
    stop = fs * window_length + fs * window_overlap * (i-1);
    feat = cleaned_data(start:stop, :);
    start = stop - fs * (window_length - window_overlap) + 1;
    all_feats(i,:) = get_features(feat, fs);
end

end