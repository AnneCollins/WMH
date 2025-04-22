% Anne Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This script performs model validation: plots the behavior across
% data-sets and models. This is used to produce figures 2B, 2C, and
% supplementary figures 10.

clear all;

%% add model code to the path
addpath('Models/')
%%
Datasets = [1 2 3 5 12 16];
dnames = {'CF12','SZ','EEG','fMRI','Dev','GL'};

% number of simulation iterations per participant. niter= 20 for the paper.
% use a smaller number (e.g. 2) for faster runtime. 
% runtime with niter=2 is 75s.
tic
niter = 20;%

% final list for paper - change model number to see validation for other
% models
ms = [9 6 2 8 26 38 42];
% ms = 81; % Model WMRLH
% ms = 92; % best RLWM + bias
% ms = 106; % best RLWM with no WM over capacity
% ms = 2;% best basic RLWM
% ms = 26; % winning WMH model
%%
for dk=[1:6]
Dataset = Datasets(dk);
dname = dnames{dk};

%%
% Define the figure size and position
figurePosition = [100, 100, 1600, 600];  % [left, bottom, width, height]

% Create the figure with the specified size and position
f1 = figure('Position', figurePosition);


%% colors
setsizeColors = [[0 0.4470 0.7410];%blue
    [0.4660 0.6740 0.1880];%green
    [0.9290 0.6940 0.1250];%yellow
    [0.8500 0.3250 0.0980];%orange
    [0.6350 0.0780 0.1840]];%red

errorColors = [[0.4940 0.1840 0.5560];[0.3010 0.7450 0.9330]];%purple, cyan
neglectColors = [[0 0 0];[.5 .5 .5]];
%%


mk = 0;
for m=ms;
    mk = mk+1;
load(['NewFits/FitRLWM_dataset',num2str(Dataset)],'Ms','All_Params')

params = All_Params{m};
%%

load(['DataSets/Expe',num2str(Dataset)])%WML_data.mat

subjects_list = unique(expe_data(:,1)');

k=0;
%% initialize empty for each dataset
LCs = [];
betaw=[];
errors = [];
neglect =[];
LCs_sim = [];
betaw_sim=[];
errors_sim = [];
neglect_sim=[];

for s = subjects_list%([4])
    k=k+1;
    %%
    % display progress
    [m k length(subjects_list)]
    % extract relevant participant data
    k=find(subjects_list==s);
    X = expe_data(expe_data(:,1)==s,:);
    % analyze and store participant data
    [LC, betas,error,erLC] = analyzeBehavior(X);
    LCs(k,:,:) = LC';
    errors(k,:,:) = error;
    neglect(k,:,:)= erLC;
    
    % run simulations
    % initialize over simulations
    iterLC=[];
    iterbetas = [];
    itererror = [];
    iterneglect = [];
    for iter = 1:niter
        % simulate data with participant parameters, experiment, and model
        eval(['sim_data = ',Ms{m}.name,'(params(k,:),X,Ms{m});'])
        % analyze and store simulated data
        [LC, betas,error,erLC] = analyzeBehavior(sim_data);
        iterLC(:,:,iter)=LC';
        itererror(:,:,iter) = error;
        iterneglect(:,:,iter)= erLC;
    end
    % average over simulations
    LC =nanmedian(iterLC,3);
    beta = nanmedian(iterbetas,3);
    % store simulations for specific participant
    LCs_sim(k,:,:) = LC;
    errors_sim(k,:,:) = nanmean(itererror,3);
    neglect_sim(k,:,:)=nanmean(iterneglect,3);
end


%% plotting
figure(f1)

%% learning curves
subplot(3,length(ms)+1,mk)
for ns=2:size(LCs,3)
errorbar(1:10,nanmean(LCs_sim(:,:,ns)),nanstd(LCs_sim(:,:,ns))/sqrt(size(LCs,1)),...
    'color',setsizeColors(ns-1,:),'linewidth',1)
hold on
errorbar(nanmean(LCs(:,:,ns)),nanstd(LCs(:,:,ns))/sqrt(size(LCs,1)),'--','linewidth',1,...
    'color',setsizeColors(ns-1,:),'linewidth',1)
ylabel('P(Cor)')
xlabel('iter')
set(gca,'fontsize',14)
end
%legend('ns2','ns3','sn4','ns5','ns6')
title([Ms{m}.ID,' - ',dname])
xlim([.5 10.5])
ylim([.2 1])
box off

%% errors
n6 = size(LCs,3);
subplot(3,length(ms)+1,mk+1*length(ms)+1)
hold on
for i = [1 2]
    errorbar(2:n6,nanmean(errors_sim(:,2:n6,i)),nanstd(errors_sim(:,2:n6,i))/sqrt(size(errors,1)),...
        'linewidth',1,'color',errorColors(i,:))
    hold on
    errorbar(2:n6,nanmean(errors(:,2:n6,i)),nanstd(errors(:,2:n6,i))/sqrt(size(errors,1)),...
        '--','linewidth',1,'color',errorColors(i,:))
end
ylabel('# prev')
%legend('chosen error','unchosen error')
title('Errors(chos/unch err)')
title('Errors')
set(gca,'fontsize',14)
xlim([1.5 6.5])
ylim([.2 1.8])


%% error difference early/late
n6 = size(LCs,3);
%figure(f2)
subplot(3,length(ms)+1,mk+2*(length(ms)+1))
hold on
for i = [1 2]
    errorbar(2:n6,nanmean(neglect_sim(:,2:n6,i)),nanstd(neglect_sim(:,2:n6,i))/sqrt(size(errors,1)),...
        'linewidth',1,'color',neglectColors(i,:))
    hold on
    errorbar(2:n6,nanmean(neglect(:,2:n6,i)),nanstd(neglect(:,2:n6,i))/sqrt(size(errors,1)),...
        '--','linewidth',1,'color',neglectColors(i,:))
end
xlabel('\Delta')
xlabel('set size')
%legend('chosen error','unchosen error')
title('Avoid Err')
set(gca,'fontsize',14)
xlim([1.5 6.5])

end

subplot(3,length(ms)+1,3*(length(ms)+1)); % Position for the legend

% Create a global legend
for i = [1 2]
    errorbar(nan,nan,nan,...
        'linewidth',1,'color',neglectColors(i,:))
    hold on
    errorbar(nan,nan,nan,...
        '--','linewidth',1,'color',neglectColors(i,:))
end
axis off; % Turn off axis for the legend subplot
box off
legend('early-Sim','early-Data','late-Sim','late-Sim','location','best')
legend('boxoff')
% filename = ['Figures/Validation-D',num2str(Dataset),'_',dname];
% savefig(f1,[filename,'.fig'])
% print([filename,'.png'], '-dpng');
end
toc