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

feats1 = getWindowedFeats(train_ecog1,1000,.10,.05);
feats2 = getWindowedFeats(train_ecog2,1000,.10,.05);
feats3 = getWindowedFeats(train_ecog3,1000,.10,.05);

%% Create R matrix
% run create_R_matrix

R1 = create_R_matrix(feats1,4);
R2 = create_R_matrix(feats2,4);
R3 = create_R_matrix(feats3,4);

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

model{1, 1} = f1;
model{2, 1} = f2;
model{3, 1} = f3;
%%
save('model.mat', 'model');


% Try at least 1 other type of machine learning algorithm, you may choose
% to loop through the fingers and train a separate classifier for angles 
% corresponding to each finger

%% Correlate data to get test accuracy and make figures (2 point)

% Calculate accuracy by correlating predicted and actual angles for each
% finger separately. Hint: You will want to use zohinterp to ensure both 
% vectors are the same length.

feats_test1 = getWindowedFeats(test_ecog1,1000,.10,.05);
feats_test2 = getWindowedFeats(test_ecog2,1000,.10,.05);
feats_test3 = getWindowedFeats(test_ecog3,1000,.10,.05);

R_test1 = create_R_matrix(feats_test1,4);
R_test2 = create_R_matrix(feats_test2,4);
R_test3 = create_R_matrix(feats_test3,4);

Y_test1 = R_test1 * f1;
Y_test2 = R_test2 * f2;
Y_test3 = R_test3 * f3;


%%
new1 = smoothdata(Y_test1, 'movmean',5);

for j = 1:5
    filtered = new1(:, j);
    for i = 1:length(filtered)
        if filtered(i) < 0.3 && filtered(i) > 0
            filtered(i) = filtered(i) * 0.1;
        elseif filtered(i) < 0
            filtered(i) = filtered(i) * .3;
        elseif filtered(i) > 1
            filtered(i) = filtered(i) * 1.25;
        end
    end
    scale1(:, j) = filtered;
end

% figure();
% plot(1:length(Y4(:)), Y4(:));
% hold on
% plot(1:length(Y4(:)), scale1(:), 'g');
% title('Subject 1 Processed Predictions vs. Test Dataglove Data');
% xlabel('Time (s)');
% ylabel('Voltage (uV)');

new2 = smoothdata(Y_test2, 'movmean',5);

for j = 1:5
    filtered = new2(:, j);
    for i = 1:length(filtered)
        if filtered(i) < 0.3 && filtered(i) > 0
            filtered(i) = filtered(i) * .5;
        elseif filtered(i) < 0
            filtered(i) = filtered(i) * 0.3;
        elseif filtered(i) > 1.5 && filtered(i) < 2
            filtered(i) = filtered(i) * 1.25;
        elseif filtered(i) > 2
            filtered(i) = filtered(i) * 1.5;
        end
    end
    scale2(:, j) = filtered;
end

% figure();
% plot(1:length(Y5(:)), Y5(:));
% hold on
% plot(1:length(Y5(:)), scale2(:), 'g');
% title('Subject 2 Processed Predictions vs. Test Dataglove Data');
% xlabel('Time (s)');
% ylabel('Voltage (uV)');

new3 = smoothdata(Y_test3, 'movmean',5);

for j = 1:5
    filtered = new3(:, j);
    for i = 1:length(filtered)
        if filtered(i) < 0.3 && filtered(i) > 0
            filtered(i) = filtered(i) * 0.1;
        elseif filtered(i) < 0
            filtered(i) = filtered(i) * .3;
        elseif filtered(i) > 1
            filtered(i) = filtered(i) * 1.25;
        end
    end
    scale3(:, j) = filtered;
end

% figure();
% plot(1:length(Y6(:)), Y6(:));
% hold on
% plot(1:length(Y6(:)), scale3(:), 'g');
% title('Subject 3 Processed Predictions vs. Test Dataglove Data');
% xlabel('Time (s)');
% ylabel('Voltage (uV)');

%%

figure();
plot(1:length(Y5(:)), Y5(:));
hold on
plot(1:length(Y5(:)), scale2(:), 'g');

%%

% Linear filter
corr1 = corr(scale1(:), Y4(:));
corr2 = corr(scale2(:), Y5(:));
corr3 = corr(scale3(:), Y6(:));

%%
load('leaderboard_data.mat');
lead1_raw = leaderboard_ecog{1};

lead2_raw = leaderboard_ecog{2};

lead3_raw = leaderboard_ecog{3};

%%
% Leaderboard linear filter

lead1 = getWindowedFeats(lead1_raw,1000,.10,.05);
lead2 = getWindowedFeats(lead2_raw,1000,.10,.05);
lead3 = getWindowedFeats(lead3_raw,1000,.10,.05);

R_lead1 = create_R_matrix(lead1,4);
R_lead2 = create_R_matrix(lead2,4);
R_lead3 = create_R_matrix(lead3,4);

Y_lead1 = R_lead1 * f1;
Y_lead2 = R_lead2 * f2;
Y_lead3 = R_lead3 * f3;

preds1_interp = interp1(1:length(Y_lead1), Y_lead1, linspace(1,length(Y_lead1),length(lead1_raw)), 'cubic');
preds2_interp = interp1(1:length(Y_lead2), Y_lead2, linspace(1,length(Y_lead2),length(lead2_raw)), 'cubic');
preds3_interp = interp1(1:length(Y_lead3), Y_lead3, linspace(1,length(Y_lead3),length(lead3_raw)), 'cubic');

%%
preds = cell(3,1);
preds{1} = preds1_interp;
preds{2} = preds2_interp;
preds{3} = preds3_interp;

% save('predicted_dg.mat', 'predicted_dg');

%%
% for sub=1:3
%     new = smoothdata(preds{sub}, 'movmean',5);
% 
%     for j = 1:5
%         filtered = new(:, j);
%         if filtered(i) < 0.3 && filtered(i) > 0
%             filtered(i) = filtered(i) * 0.1;
%         elseif filtered(i) < 0
%             filtered(i) = filtered(i) * .3;
%         elseif filtered(i) > 1
%             filtered(i) = filtered(i) * 1.25;
%         end
%         scaled_lb(:, j) = filtered;
%     end
%     predicted_dg{sub, 1} = scaled_lb;
% end

%%
new_lb1 = smoothdata(preds{1}, 'movmean',5);

for j = 1:5
    filtered = new_lb1(:, j);
    for i = 1:length(filtered)
        if filtered(i) < 0.3 && filtered(i) > 0
            filtered(i) = filtered(i) * 0.1;
        elseif filtered(i) < 0
            filtered(i) = filtered(i) * .3;
        elseif filtered(i) > 1
            filtered(i) = filtered(i) * 1.25;
        end
    end
    scale_lb1(:, j) = filtered;
end


new_lb2 = smoothdata(preds{2}, 'movmean',5);

for j = 1:5
    filtered = new_lb2(:, j);
    for i = 1:length(filtered)
        if filtered(i) < 0.3 && filtered(i) > 0
            filtered(i) = filtered(i) * .5;
        elseif filtered(i) < 0
            filtered(i) = filtered(i) * 0.3;
        elseif filtered(i) > 1.5 && filtered(i) < 2
            filtered(i) = filtered(i) * 1.25;
        elseif filtered(i) > 2
            filtered(i) = filtered(i) * 1.5;
        end
    end
    scale_lb2(:, j) = filtered;
end


new_lb3 = smoothdata(preds{3}, 'movmean',5);

for j = 1:5
    filtered = new_lb3(:, j);
    for i = 1:length(filtered)
        if filtered(i) < 0.3 && filtered(i) > 0
            filtered(i) = filtered(i) * 0.1;
        elseif filtered(i) < 0
            filtered(i) = filtered(i) * .3;
        elseif filtered(i) > 1
            filtered(i) = filtered(i) * 1.25;
        end
    end
    scale_lb3(:, j) = filtered;
end

predicted_dg{1, 1} = scale_lb1;
predicted_dg{2, 1} = scale_lb2;
predicted_dg{3, 1} = scale_lb3;

%%
% save('predicted_dg.mat', 'predicted_dg');

