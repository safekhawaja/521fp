function [R]=create_R_matrix_release(features, N_wind)
    %
    % get_features_release.m
    %
    % Instructions: Write a function to calculate R matrix.             
    %
    % Input:    features:   (samples x (channels*features))
    %           N_wind:     Number of windows to use
    %
    % Output:   R:          (samples x (N_wind*channels*features))
    % 
%% Your code here (5 points)

R = zeros(length(features), N_wind * size(features, 2));

vec1 = features(1:N_wind - 1, :);
features = [vec1; features];

for i = 1:length(features)
    if N_wind + i - 1 <= size(features, 1)
        feats = features(i:N_wind + i - 1, :).';
        R(i,:) = reshape(feats, 1, []);
    else
        break;
    end
end

col1 = ones(size(R,1),1);
R = [col1 R];

end