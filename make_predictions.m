function [predicted_dg] = make_predictions(test_ecog)

% INPUTS: test_ecog - 3 x 1 cell array containing ECoG for each subject, where test_ecog{i} 
% to the ECoG for subject i. Each cell element contains a N x M testing ECoG,
% where N is the number of samples and M is the number of EEG channels.

% OUTPUTS: predicted_dg - 3 x 1 cell array, where predicted_dg{i} contains the 
% data_glove prediction for subject i, which is an N x 5 matrix (for
% fingers 1:5)

% Run time: The script has to run less than 1 hour. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lead1_raw = test_ecog{1};
lead2_raw = test_ecog{2};
lead3_raw = test_ecog{3};

load('model.mat');
f1 = model{1};
f2 = model{2};
f3 = model{3};

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

preds = cell(3,1);
preds{1} = preds1_interp;
preds{2} = preds2_interp;
preds{3} = preds3_interp;

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

predicted_dg = cell(3,1);
predicted_dg{1, 1} = scale_lb1;
predicted_dg{2, 1} = scale_lb2;
predicted_dg{3, 1} = scale_lb3;

save('predicted_dg.mat', 'predicted_dg');

end

