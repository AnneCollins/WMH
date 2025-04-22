% Anne GE Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This script plots the behavior across 6 data-sets. This is included in
% figure 1. 

%%
clear all;

% Define the figure size and position
figurePosition = [100, 100, 1600, 600];  % [left, bottom, width, height]

% Create the figure with the specified size and position
f1 = figure('Position', figurePosition);

% 6 data sets to consider
ds = [1 2 3 5 12 16];
% dataset names
dnames = {'CF12','SZ','EEG','fMRI','Dev','GL'};

%% define color schemecolors
setsizeColors = [[0 0.4470 0.7410];%blue
    [0.4660 0.6740 0.1880];%green
    [0.9290 0.6940 0.1250];%yellow
    [0.8500 0.3250 0.0980];%orange
    [0.6350 0.0780 0.1840]];%red

errorColors = [[0.4940 0.1840 0.5560];[0.3010 0.7450 0.9330]];%purple, cyan
neglectColors = [[0 0 0];[.5 .5 .5]];

%% plot each dataset
kd=0;
for Dataset = ds% loop over data sets
    
    %% load the data set
    kd = kd+1;
    ParticipantsToExclude=[];
    load(['DataSets/Expe',num2str(Dataset)])%WML_data.mat
    dname = dnames{kd};
    
    %% extract the list of participants
    subjects_list = setdiff(unique(expe_data(:,1)'),ParticipantsToExclude);
    
    k=0;
    %% initialize variables
    LCs = [];% learning curves
    betaw = [];%logistic regression weights
    errors = [];% error analyzis
    neglect = [];% error avoidance neglect
    %% loop over subjects
    for s = subjects_list%([4])
        k=k+1;
        %% extract participant data
        %[k length(subjects_list)]
        k=find(subjects_list==s);
        X = expe_data(expe_data(:,1)==s,:);
        
        % do participant specific analysis
        [LC, betas,error,erLC] = analyzeBehavior(X);
        % store summary statistics for participants
        LCs(k,:,:) = LC';
        betaw(k,:,:)=1./(1+exp(-betas));
        errors(k,:,:) = error;
        neglect(k,:,:)= erLC;
    end
    
    %% stats - t-tests
    disp(['Stats for ',dname])
    errorEffect = errors(:,2:end,1)-errors(:,2:end,2);
    [h,p,ci,stats] = ttest(neglect(:,end,2));
    disp(['Effect of errors type ns6 late: p=',num2str(p)])
    disp(['effect of errors type ns6 late: t(',num2str(stats.df),')=',num2str(stats.tstat)])
    
    %% stats - mixed effects of set size
    perf =squeeze(nanmean(LCs(:,:,2:end),2));
    
    IDs = repmat((1:size(perf,1))',[1,size(perf,2)]);
    setsizes = repmat((2:size(LCs,3)),[size(perf,1),1]);
    data = table(IDs(:), perf(:), setsizes(:), errorEffect(:),'VariableNames', {'ParticipantID', 'Performance', 'SetSize','DeltaError'});
    % uncomment to run relevant models
    % Set size on performance
    % disp('Performance ~ SetSize + (1|ParticipantID)')
    % lme = fitlme(data, 'Performance ~ SetSize + (1|ParticipantID)');
    % disp(lme.Coefficients);
    % Set size on error difference
    % disp('DeltaError ~ SetSize + (1|ParticipantID)')
    % lme = fitlme(data, 'DeltaError ~ SetSize + (1|ParticipantID)');
    % disp(lme.Coefficients);
    
    %% plotting - LCs
    for ns=2:size(LCs,3)
        %subplot(length(ds),3,1+3*(kd-1) )
        subplot(3,length(ds),kd)
        hold on
        errorbar(nanmean(LCs(:,:,ns)),nanstd(LCs(:,:,ns))/sqrt(size(LCs,1)),'linewidth',1,...
            'color',setsizeColors(ns-1,:))
        ylabel('P(Cor)')
        xlabel('iter')
        set(gca,'fontsize',14)
    end
    if Dataset==ds(end)
        legend('ns2','ns3','sn4','ns5','ns6')
    end
    title([dname,' - N=',num2str(k)])
    xlim([0.5 10.5])
    ylim([.2 1])
    
    maxns = size(errors,2);
    %% plotting - error pattern
    subplot(3,length(ds),kd+length(ds))
    hold on
    for i = [1 2]
        hold on
        errorbar(2:maxns,nanmean(errors(:,2:maxns,i)),nanstd(errors(:,2:maxns,i))/sqrt(size(errors,1)),...
            'linewidth',1,'color',errorColors(i,:))
    end
    ylabel('# previous')
    if Dataset==ds(end)
        legend('chosen error','unchosen error')
    end
    set(gca,'fontsize',14)
    xlim([1.5 6.5])
    ylim([.2 1.8])
    %% plot avoid error neglect
    subplot(3,length(ds),kd+2*length(ds))
    hold on
    for i = [1 2]
        hold on
        errorbar(2:maxns,nanmean(neglect(:,2:maxns,i)),nanstd(neglect(:,2:maxns,i))/sqrt(size(errors,1)),...
            '--','linewidth',1,'color',neglectColors(i,:))
    end
    xlabel('\Delta')
    xlabel('set size')
    if Dataset==ds(end)
    legend('early','late')
    end
    title('Avoid Err')
    set(gca,'fontsize',14)
    xlim([1.5 6.5])
    
    %%
    
    clear LCs betaw errors
    
end
