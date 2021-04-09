function [features] = get_features_release(clean_data,fs)
    %
    % get_features_release.m
    %
    % Instructions: Write a function to calculate features.
    %               Please create 4 OR MORE different features for each channel.
    %               Some of these features can be of the same type (for example, 
    %               power in different frequency bands, etc) but you should
    %               have at least 2 different types of features as well
    %               (Such as frequency dependent, signal morphology, etc.)
    %               Feel free to use features you have seen before in this
    %               class, features that have been used in the literature
    %               for similar problems, or design your own!
    %
    % Input:    clean_data: (samples x channels)
    %           fs:         sampling frequency
    %
    % Output:   features:   (1 x (channels*features))
    % 
%% Your code here (8 points)
LLFn = @(x) sum(abs(diff(x)));
Area = @(x) sum(abs(x));
Energy = @(x) sum(x.^2);
ZC = @(x) sum(abs(diff((x-mean(x))>0)));
LMP = @(x) mean(x);

features = [LMP(clean_data) bandpower(clean_data,fs,[8 12]) bandpower(clean_data,fs,[18 24]) bandpower(clean_data,fs,[75 115]) bandpower(clean_data,fs,[125 159]) bandpower(clean_data,fs,[159 175])];
%features = [LLFn(clean_data) Area(clean_data) sum(diff(sign(diff(clean_data)))~=0)];
end

