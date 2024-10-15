% Anne Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This script performs model comparison. It is used for figures 2A, S6 and
% s7.


clear all

cd ..
SPMPackages = [pwd,'/spm12/'];
addpath(genpath(SPMPackages));
cd RLWM

%% models to consider and resorting
% better ordering of models
remapping = [1 4 10 2 6 7 3 8 5 9];
remapping = [remapping 10+remapping 20+remapping 31:40 40+remapping ...
    50+remapping 60+remapping 70+remapping];

ticnames =[];
%% for supplementary figure 6, select models 1:10
toconsider =1:10;
%% for figure 2A, select models below;
toconsider = [9 6 2 8 26 38 42];% list for the first paper submission.
%% Supplementary figure with free WM alpha+ or RL forgetting
% toconsider = [2 8 52 62 68];

%% order models for plotting.

sorted_toconsider = [];
for m = remapping%(1:10)
    if find(toconsider==m)
        sorted_toconsider = [sorted_toconsider m];
        toconsider(toconsider==m)=[];
    end
end
toconsider = sorted_toconsider;

%% load fits
load(['Fits/FitRLWM_dataset1'])
for i=1:length(toconsider)
    ticnames{i} = Ms{toconsider(i)}.ID;
end
%% run over datasets

ds = [1 2 3 5 12 16];% 51 2 2
dnames = {'CF12','SZ','EEG','fMRI','Dev','GL','Imp'};
figurePosition = [100, 100, 800, 400];  % [left, bottom, width, height]
f1=figure('position',figurePosition);
count=0;
for dataset = ds([1:6])
    count=count+1;
    load(['Fits/FitRLWM_dataset',num2str(dataset)])
    %% Compare baseline models
    
    AICs = squeeze(All_fits(:,3,:));
    mAICs =AICs - repmat(mean(AICs(:,toconsider),2),1,size(AICs,2));
    
    %% compute exceedence probability
    [alpha,exp_r,xp,pxp,bor] = spm_BMS(-AICs(:,toconsider));
    [ma,mi]=max(pxp);
    subplot(2,3,count)
    bar(exp_r)
    box off
    ylim([0 .75])
    title([dnames{count},'; pxp(best)=',num2str(ma)])
    set(gca,'xtick',1:length(toconsider),'xticklabels',ticnames,'fontsize',12)
    ylabel('exp_r')
    xtickangle(45)
end
