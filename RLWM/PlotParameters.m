clear all 
% Anne Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

%% Plots Fig. S11 (fit model parameters)


toconsider = [9 6 2 8 26 38 42];


ds = [1 2 3 5 12 16];% 51 2 2
figurePosition = [100, 100, 1200, 1200];  % [left, bottom, width, height]
f1=figure('position',figurePosition);
dnames = {'CF12','SZ','EEG','fMRI','Dev','GL','Imp'};
i=0;
maxp=8;
nm=length(toconsider);
for m = toconsider
    i=i+1;
count=0;
hold on
for dataset = ds([1:6])
    count=count+1;
    %load(['Fits/FitRLWM_dataset',num2str(dataset)])
    load(['NewFits/FitRLWM_dataset',num2str(dataset)])
    Params = All_Params{m};    
    n = size(Params,1);
    np = size(Params,2);
    for p=1:np
        pnames = Ms{m}.pnames;
        par_name = pnames{find(Ms{m}.thetamapping==p,1)};
        subplot(nm,maxp,p+(i-1)*maxp)
        hold on
        plot((1:n)/n,sort(Params(:,p)),'.-')
        ylabel(par_name)
        title(Ms{m}.ID)
    end
end
legend(dnames)
end