% Anne Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This script performs model validation: plots the behavior across
%  models. This is used to produce figure 3B.

clear all;
Dataset = 14;
dname = '14';%dnames{dk};
addpath('Models/')

%%
% Define the figure size and position
figurePosition = [100, 100, 600, 600];  % [left, bottom, width, height]

% Create the figure with the specified size and position
f1 = figure('Position', figurePosition);

niter = 20;% a few minutes with 20 iterations, 10-20s with 2 iterations per participant
ms = [3 6 7 16];% models to consider

%% colors
setsizeColors = [[0 0.4470 0.7410];%blue
    [0.4660 0.6740 0.1880];%green
    [0.9290 0.6940 0.1250];%yellow
    [0.8500 0.3250 0.0980];%orange
    [0.6350 0.0780 0.1840]];%red

errorColors = [[0.4940 0.1840 0.5560];[0.3010 0.7450 0.9330];[.5 .5 .5];[.5 .5 .5];[.5 .5 .5];[.5 .5 .5]];%purple, cyan
neglectColors = [[0 0 0];[.5 .5 .5]];
%%


mk = 0;
for m=ms;
    mk = mk+1;
    load(['Fits/FitRL2_dataset',num2str(Dataset)])
    
    params = All_Params{m};
    %%
    
    load(['DataSets/Expe',num2str(Dataset)])%WML_data.mat
    
    subjects_list = unique(expe_data(:,1)');
    
    k=0;
    %%
    LCs = [];
    LCs_sim = [];
    
    for s = subjects_list%([4])
        k=k+1;
        %%
        [m k length(subjects_list)]
        k=find(subjects_list==s);
        X = expe_data(expe_data(:,1)==s,:);
        
        [LC] = analyzeBehavior_RLWMP(X);
        LCs(k,:,:) = LC';
        iterLC=[];
        iterbetas = [];
        for iter = 1:niter
            sim_data = RL2(params(k,:),X,Ms{m});
            
            [LC] = analyzeBehavior_RLWMP(sim_data);
            iterLC(:,:,iter)=LC';
        end
        LC =nanmedian(iterLC,3);
        beta = nanmedian(iterbetas,3);
        LCs_sim(k,:,:) = LC;
    end
    
    
    %% plotting
    figure(f1)
    % LCs = LCs_sim
    % betaw = betaw_sim
    for ns=[3 6]
        subplot(2,2,mk)
        errorbar(1:10,nanmean(LCs_sim(:,:,ns)),nanstd(LCs_sim(:,:,ns))/sqrt(size(LCs,1)),...
            'color',setsizeColors(ns-1,:),'linewidth',2)
        hold on
        errorbar(nanmean(LCs(:,:,ns)),nanstd(LCs(:,:,ns))/sqrt(size(LCs,1)),...
            '--','color',setsizeColors(ns-1,:),'linewidth',2)
        ylabel('P(Cor)')
        xlabel('iter')
        set(gca,'fontsize',14)
    end
    %legend('ns2','ns3','sn4','ns5','ns6')
    title([Ms{m}.ID])
    xlim([.5 10.5])
    box off
end
legend({'ns3 model','ns3 data','ns6 model','ns6 data'},'location','southeast')

filename = ['Figures/Validation-Ms',num2str(ms),'-D',num2str(Dataset),'_',dname];
%UNCOMMENT TO SAVE FILES
% savefig(f1,[filename,'.fig'])
% print([filename,'.png'], '-dpng');
% print([filename,'.svg'], '-dsvg');

