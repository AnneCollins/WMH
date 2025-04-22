% Anne Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This script performs model comparison. It is used for figures 3A
% it also plots fitted parameters in fig. S12.

clear all

cd ..
SPMPackages = [pwd,'/spm12/'];
addpath(genpath(SPMPackages));
cd RLWMP
%note that the spm package may need to be installed from https://www.fil.ion.ucl.ac.uk/spm/software/

%% models to consider and resorting
n=19;
remapping = [1:n];

ticnames =[];
toconsider = [1:2 6 5 3:4 7 16];%
% toconsider = [1:7 16];
% 
% sorted_toconsider = [];
% for m = remapping%(1:10)
%     if find(toconsider==m)
%         sorted_toconsider = [sorted_toconsider m];
%         toconsider(toconsider==m)=[];
%     end
% end
% toconsider = sorted_toconsider;

    load(['Fits/FitRL2_dataset14'])
for i=1:length(toconsider)
ticnames{i} = Ms{toconsider(i)}.ID;
end
%%

ds = 14;
figure%(2)
count=0;
for dataset = ds
    count=count+1;
    load(['Fits/FitRL2_dataset',num2str(dataset)])

%% Compare baseline models

AICs = squeeze(All_fits(:,3,:));
mAICs =AICs - repmat(mean(AICs(:,toconsider),2),1,size(AICs,2));
% BICs = squeeze(All_fits(:,4,:));
% mBICs =BICs - repmat(mean(BICs(:,toconsider),2),1,size(BICs,2));

% other visualization
% figure(1)
% subplot(1,length(ds),count)
% hold on
% bar(mean(mAICs(:,toconsider)))
% plot(mAICs(:,toconsider)','ko-')
% errorbar(mean(mAICs(:,toconsider)),std(mAICs(:,toconsider))/sqrt(size(mAICs,1)),'linewidth',2)
% set(gca,'xtick',1:length(toconsider),'xticklabels',ticnames,'fontsize',14)
% ylabel('\Delta AIC')
% xtickangle(45)

%%
% [alpha,exp_r,xp,pxp,bor] = spm_BMS(-AICs(:,toconsider));
% [ma,mi]=max(pxp);
% %subplot(3,2,count)
% bar(exp_r,'facecolor',[.5 .5 .5])
% title(['Dataset ',num2str(dataset),'; pxp(best)=',num2str(ma)])
% set(gca,'xtick',1:length(toconsider),'xticklabels',ticnames,'fontsize',14)
% ylabel('exp_r')
% xtickangle(45)
% box off

  %% mean AICs
    
    [~,best]= min(mean(mAICs(:,toconsider))); 
    mAICs = mAICs(:,toconsider)-repmat(mAICs(:,toconsider(best)),1,length(toconsider));
    subplot(1,2,1)
    bar(mean(mAICs),'facecolor',[.5 .5 .5])
    hold on
    errorbar(mean(mAICs), std(mAICs)/sqrt(size(mAICs,1)),'.k')
    box off
    %ylim([0 .75])
    title(['best=',ticnames{best}])
    set(gca,'xtick',1:length(toconsider),'xticklabels',ticnames,'fontsize',12)
    ylabel('\Delta AIC')
    xtickangle(45)
    
    %% Proportion best fit. 
    [~,best]= min(mAICs,[],2);
    for m=1:length(toconsider)
        prop(m)=mean(best==m);
    end
    subplot(1,2,2)
    bar(prop,'facecolor',[.5 .5 .5])
    box off
    [~,bestmodel] = max(prop);
    title(['best=',ticnames{bestmodel}])
    set(gca,'xtick',1:length(toconsider),'xticklabels',ticnames,'fontsize',12)
    ylabel('Proportion best fit')
    xtickangle(45)
end

%% load parameters
 
clear all

m = 7;
dataset=14;
figure;
load(['Fits/FitRL2_dataset',num2str(dataset)])
Params = All_Params{m};
%% plot params (Fig. S12)
lParams = Params;
% transform learning rates for visualazation
lParams(:,1:4)=Params(:,1:4).^(1/2);
% reparameterize WM weights for ns6

lParams(:,6) = lParams(:,6).*lParams(:,5);
figure;
% create x-axis noise for visualization.
x=repmat(1:2,size(lParams,1),1)+.05*randn(size(lParams,1),2);
% neg learning rates
subplot(3,2,1)
plot(x,lParams(:,[1 3]),'ok')
hold on
errorbar(mean(lParams(:,[1 3])),std(lParams(:,[1 3]))/sqrt(size(lParams,1)),'+r','linewidth',2)
set(gca,'fontsize',14,'xtick',1:2,'xticklabels',{Ms{m}.param_names{1},Ms{m}.param_names{3}})
ylim([0,1])
ylabel('\alpha ^{.5}')
box off

% pos learning rates
subplot(3,2,2)
plot(x,lParams(:,[2 4]),'ok')
hold on
errorbar(mean(lParams(:,[2 4])),std(lParams(:,[2 4]))/sqrt(size(lParams,1)),'+r','linewidth',2)
set(gca,'fontsize',14,'xtick',1:2,'xticklabels',{Ms{m}.param_names{2},Ms{m}.param_names{4}})
ylim([0,1])
box off

% H learning rates
subplot(3,2,3)
plot(x,lParams(:,[1 2]),'ok')
hold on
errorbar(mean(lParams(:,[1 2])),std(lParams(:,[1 2]))/sqrt(size(lParams,1)),'+r','linewidth',2)
set(gca,'fontsize',14,'xtick',1:2,'xticklabels',{Ms{m}.param_names{1},Ms{m}.param_names{2}})
ylim([0,1])
ylabel('\alpha ^{.5}')
box off

% WM learning rates
subplot(3,2,4)
plot(x,lParams(:,[3 4]),'ok')
hold on
errorbar(mean(lParams(:,[3 4])),std(lParams(:,[3 4]))/sqrt(size(lParams,1)),'+r','linewidth',2)
set(gca,'fontsize',14,'xtick',1:2,'xticklabels',{Ms{m}.param_names{3},Ms{m}.param_names{4}})
ylim([0,1])
box off

% rhos
subplot(3,2,5)
plot(x,lParams(:,[5:6]),'ok')
hold on
errorbar(mean(lParams(:,[5 6])),std(lParams(:,[5 6]))/sqrt(size(lParams,1)),'+r','linewidth',2)
set(gca,'fontsize',14,'xtick',1:2,'xticklabels',{Ms{m}.param_names{5},Ms{m}.param_names{6}})
ylim([0,1])
box off

% other params
x(:,3) = 3+.05*randn(size(lParams,1),1);
subplot(3,2,6)
plot(x,lParams(:,[7:9]),'ok')
hold on
errorbar(mean(lParams(:,[7:9])),std(lParams(:,[7:9]))/sqrt(size(lParams,1)),'+r','linewidth',2)
set(gca,'fontsize',14,'xtick',1:3,'xticklabels',{Ms{m}.param_names{7},Ms{m}.param_names{8},Ms{m}.param_names{9}})
box off

