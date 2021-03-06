%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file show the confidence cues computed from C++ code
% Input files:
%  - cost1.txt cost2.txt : minimum and second minimum cost
%  - costMLM.txt : MLM confidence
%  - costLC.txt : LC confidence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

ImageID = '01';
%% PKRN
cost1 = load(['cost1_' ImageID '.txt']);
cost2 = load(['cost2_' ImageID '.txt']);

small_num = 100;
confid = (cost2+small_num)./(cost1+small_num)-1;
figure;imagesc(fliplr(confid),[0,1]);colormap(gray);axis equal
axis off
title('PKRN')

%% MLM
costMLM = load(['costMLM_' ImageID '.txt']);
figure;imagesc(fliplr(costMLM));colormap(gray);axis equal
axis off
title('MLM')

%% LC
costLC = load(['costLC_' ImageID '.txt']);
figure;imagesc(fliplr(costLC));colormap(gray);axis equal
axis off
title('LC')