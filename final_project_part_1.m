%% Final project part 1
% Prepared by John Bernabei and Brittany Scheid

% One of the oldest paradigms of BCI research is motor planning: predicting
% the movement of a limb using recordings from an ensemble of cells involved
% in motor control (usually in primary motor cortex, often called M1).

% This final project involves predicting finger flexion using intracranial EEG (ECoG) in three human
% subjects. The data and problem framing come from the 4th BCI Competition. For the details of the
% problem, experimental protocol, data, and evaluation, please see the original 4th BCI Competition
% documentation (included as separate document). The remainder of the current document discusses
% other aspects of the project relevant to BE521.


%% Start the necessary ieeg.org sessions (0 points)

% username = 'your_username';
% passPath = 'your_ieeglogin.bin';

% Load training ecog from each of three patients
% s1_train_ecog = IEEGSession('I521_Sub1_Training_ecog', username, passPath);

% Load training dataglove finger flexion values for each of three patients
% s1_train_dg = IEEGSession('I521_Sub1_Training_dg', username, passPath);

%% Extract dataglove and ECoG data 
% Dataglove should be (samples x 5) array 
% ECoG should be (samples x channels) array

% Split data into a train and test set (use at least 50% for training)

load('raw_training_data.mat');

split = .75 * length(train_dg{1});

train_ecog1 = train_ecog{1}(1:split,:);
train_ecog2 = train_ecog{2}(1:split,:);
train_ecog3 = train_ecog{3}(1:split,:);

train_dg1 = train_dg{1}(1:split,:);
train_dg2 = train_dg{2}(1:split,:);
train_dg3 = train_dg{3}(1:split,:);

test_ecog1 = train_ecog{1}(split+1:end,:);
test_ecog2 = train_ecog{2}(split+1:end,:);
test_ecog3 = train_ecog{3}(split+1:end,:);

test_dg1 = train_dg{1}(split+1:end,:);
test_dg2 = train_dg{2}(split+1:end,:);
test_dg3 = train_dg{3}(split+1:end,:);

%% Get Features
% run getWindowedFeats_release function

feats1 = getWindowedFeats(train_ecog1,1000,.100,.050);
feats2 = getWindowedFeats(train_ecog2,1000,.100,.050);
feats3 = getWindowedFeats(train_ecog3,1000,.100,.050);

%% Create R matrix
% run create_R_matrix

R1 = create_R_matrix(feats1,3);
R2 = create_R_matrix(feats2,3);
R3 = create_R_matrix(feats3,3);

%% Train classifiers (8 points)


% Classifier 1: Get angle predictions using optimal linear decoding. That is, 
% calculate the linear filter (i.e. the weights matrix) as defined by 
% Equation 1 for all 5 finger angles.

Y1 = zeros(4500, 5);
Y2 = zeros(4500, 5);
Y3 = zeros(4500, 5);
Y4 = zeros(1500, 5);
Y5 = zeros(1500, 5);
Y6 = zeros(1500, 5);

for i=1:5
    Y1(:, i) = decimate(train_dg1(:, i),50);
    Y2(:, i) = decimate(train_dg2(:, i),50);
    Y3(:, i) = decimate(train_dg3(:, i),50);
    Y4(:, i) = decimate(test_dg1(:, i),50);
    Y5(:, i) = decimate(test_dg2(:, i),50);
    Y6(:, i) = decimate(test_dg3(:, i),50);
end
Y1(1,:) = [];
Y2(1,:) = [];
Y3(1,:) = [];
Y4(1,:) = [];
Y5(1,:) = [];
Y6(1,:) = [];

% f1 = mldivide(R1' * Y1, R1' * R1);
% f2 = mldivide(R2' * Y2, R2' * R2);
% f3 = mldivide(R3' * Y3, R3' * R3);

f1 = mldivide(R1' * R1, R1' * Y1);
f2 = mldivide(R2' * R2, R2' * Y2);
f3 = mldivide(R3' * R3, R3' * Y3);

% Try at least 1 other type of machine learning algorithm, you may choose
% to loop through the fingers and train a separate classifier for angles 
% corresponding to each finger
%%
feats1_std = (feats1-mean(feats1))./std(feats1);
feats2_std = (feats2-mean(feats2))./std(feats2);
feats3_std = (feats3-mean(feats3))./std(feats3);

feats1_test = getWindowedFeats(test_ecog1,1000,.100,.050);
feats2_test = getWindowedFeats(test_ecog2,1000,.100,.050);
feats3_test = getWindowedFeats(test_ecog3,1000,.100,.050);


feats1_test_std = (feats1_test-mean(feats1_test))./std(feats1_test);
feats2_test_std = (feats2_test-mean(feats2_test))./std(feats2_test);
feats3_test_std = (feats3_test-mean(feats3_test))./std(feats3_test);

% feats1_std = feats1;
% feats1_test_std = getWindowedFeats(test_ecog1,1000,.100,.050);
% 
% feats2_std = feats2;
% feats2_test_std = getWindowedFeats(test_ecog2,1000,.100,.050);
% 
% feats3_std = feats3;
% feats3_test_std = getWindowedFeats(test_ecog3,1000,.100,.050);

%%

% Subject 1
preds1 = zeros(1499,5);

model_11 = fitrlinear(feats1_std,Y1(:,1));
preds1(:,1) = predict(model_11,feats1_test_std);

model_12 = fitrlinear(feats1_std,Y1(:,2));
preds1(:,2) = predict(model_12,feats1_test_std);

model_13 = fitrlinear(feats1_std,Y1(:,3));
preds1(:,3) = predict(model_13,feats1_test_std);

model_14 = fitrlinear(feats1_std,Y1(:,4));
preds1(:,4) = predict(model_14,feats1_test_std);

model_15 = fitrlinear(feats1_std,Y1(:,5));
preds1(:,5) = predict(model_15,feats1_test_std);

% Subject 2
preds2 = zeros(1499,5);
model_21 = fitrlinear(feats2_std,Y2(:,1));
preds2(:,1) = predict(model_21,feats2_test_std);

model_22 = fitrlinear(feats2_std,Y2(:,2));
preds2(:,2) = predict(model_22,feats2_test_std);

model_23 = fitrlinear(feats2_std,Y2(:,3));
preds2(:,3) = predict(model_23,feats2_test_std);

model_24 = fitrlinear(feats2_std,Y2(:,4));
preds2(:,4) = predict(model_24,feats2_test_std);

model_25 = fitrlinear(feats2_std,Y2(:,5));
preds2(:,5) = predict(model_25,feats2_test_std);

% Subject 3
preds3 = zeros(1499,5);
model_31 = fitrlinear(feats3_std,Y3(:,1));
preds3(:,1) = predict(model_31,feats3_test_std);

model_32 = fitrlinear(feats3_std,Y3(:,2));
preds3(:,2) = predict(model_32,feats3_test_std);

model_33 = fitrlinear(feats3_std,Y3(:,3));
preds3(:,3) = predict(model_33,feats3_test_std);

model_34 = fitrlinear(feats3_std,Y3(:,4));
preds3(:,4) = predict(model_34,feats3_test_std);

model_35 = fitrlinear(feats3_std,Y3(:,5));
preds3(:,5) = predict(model_35,feats3_test_std);

% Try a form of either feature or prediction post-processing to try and
% improve underlying data or predictions.

%% Correlate data to get test accuracy and make figures (2 point)

% Calculate accuracy by correlating predicted and actual angles for each
% finger separately. Hint: You will want to use zohinterp to ensure both 
% vectors are the same length.

feats_test1 = getWindowedFeats(test_ecog1,1000,.100,.050);
feats_test2 = getWindowedFeats(test_ecog2,1000,.100,.050);
feats_test3 = getWindowedFeats(test_ecog3,1000,.100,.050);

R_test1 = create_R_matrix(feats_test1,3);
R_test2 = create_R_matrix(feats_test2,3);
R_test3 = create_R_matrix(feats_test3,3);

Y_test1 = R_test1 * f1;
Y_test2 = R_test2 * f2;
Y_test3 = R_test3 * f3;


%%
% Linear filter
corr1 = corr(Y_test1(:), Y4(:));
corr2 = corr(Y_test2(:), Y5(:));
corr3 = corr(Y_test3(:), Y6(:));
%%
% kNN models
corr_model1 = corr(preds1(:), Y4(:));
corr_model2 = corr(preds2(:), Y5(:));
corr_model3 = corr(preds3(:), Y6(:));

%%

lead1_raw = leaderboard_ecog{1};
lead1 = getWindowedFeats(lead1_raw,1000,.100,.050);
lead1_std = (lead1-mean(lead1))./std(lead1);
%lead1_std = lead1;

lead2_raw = leaderboard_ecog{2};
lead2 = getWindowedFeats(lead2_raw,1000,.100,.050);
lead2_std = (lead2-mean(lead2))./std(lead2);
%lead2_std = lead2;

lead3_raw = leaderboard_ecog{3};
lead3 = getWindowedFeats(lead3_raw,1000,.100,.050);
lead3_std = (lead3-mean(lead3))./std(lead3);
%lead3_std = lead3;

% Subject 1
preds1 = zeros(length(lead1_std),5);
preds1(:,1) = predict(model_11,lead1_std);
preds1(:,2) = predict(model_12,lead1_std);
preds1(:,3) = predict(model_13,lead1_std);
preds1(:,4) = predict(model_14,lead1_std);
preds1(:,5) = predict(model_15,lead1_std);
preds1_interp = interp1(1:length(preds1), preds1, linspace(1,length(preds1),length(lead1_raw)), 'cubic');

% Subject 2
preds2 = zeros(length(lead2_std),5);
preds2(:,1) = predict(model_21,lead2_std);
preds2(:,2) = predict(model_22,lead2_std);
preds2(:,3) = predict(model_23,lead2_std);
preds2(:,4) = predict(model_24,lead2_std);
preds2(:,5) = predict(model_25,lead2_std);
preds2_interp = interp1(1:length(preds2), preds2, linspace(1,length(preds2),length(lead2_raw)), 'cubic');

% Subject 3
preds3 = zeros(length(lead3_std),5);
preds3(:,1) = predict(model_31,lead3_std);
preds3(:,2) = predict(model_32,lead3_std);
preds3(:,3) = predict(model_33,lead3_std);
preds3(:,4) = predict(model_34,lead3_std);
preds3(:,5) = predict(model_35,lead3_std);
preds3_interp = interp1(1:length(preds3), preds3, linspace(1,length(preds3),length(lead3_raw)), 'cubic');

predicted_dg = cell(3,1);
predicted_dg{1} = preds1_interp;
predicted_dg{2} = preds2_interp;
predicted_dg{3} = preds3_interp;

save('predicted_dg.mat', 'predicted_dg');

%%
% Leaderboard linear filter

lead1 = getWindowedFeats(lead1_raw,1000,.100,.050);
lead2 = getWindowedFeats(lead2_raw,1000,.100,.050);
lead3 = getWindowedFeats(lead3_raw,1000,.100,.050);

R_lead1 = create_R_matrix(lead1,3);
R_lead2 = create_R_matrix(lead2,3);
R_lead3 = create_R_matrix(lead3,3);

Y_lead1 = R_lead1 * f1;
Y_lead2 = R_lead2 * f2;
Y_lead3 = R_lead3 * f3;

preds1_interp = interp1(1:length(Y_lead1), Y_lead1, linspace(1,length(Y_lead1),length(lead1_raw)), 'cubic');
preds2_interp = interp1(1:length(Y_lead2), Y_lead2, linspace(1,length(Y_lead2),length(lead2_raw)), 'cubic');
preds3_interp = interp1(1:length(Y_lead3), Y_lead3, linspace(1,length(Y_lead3),length(lead3_raw)), 'cubic');


predicted_dg = cell(3,1);
predicted_dg{1} = preds1_interp;
predicted_dg{2} = preds2_interp;
predicted_dg{3} = preds3_interp;

save('predicted_dg.mat', 'predicted_dg');

