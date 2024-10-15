% Anne Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This script performs Model and parameter identification. 
% This script is ***slow*** to run (multiple hours) if runGenRec=true. 

% The last two cells of the script are fast to run and generate Figs. S8
% and S9 based on saved output of the previous cells. 
% If runGenRec=false, only those cells are run.

% this script generated all the data structures in the GenRec Folder.

%% 
runGenRec = false;% make it true to re-run the generate and recover (SLOW)
%%
if runGenRec

clear all

%% define data set and models to consider
Datasets = [1 2 3 5 12 16];
dk=5;
Dataset = Datasets(dk);
dnames = {'CF12','SZ','EEG','fMRI','Dev','GL'};
dname = dnames{dk};
ms = [8 26];%
% ms = [26 38];
% ms = [8 38];

%% load data and subjects and fit params

load(['DataSets/Expe',num2str(Dataset)])%WML_data.mat
load(['Fits/FitRLWM_dataset',num2str(Dataset)],'Ms','All_Params')


subjects_list = unique(expe_data(:,1)');
nsim = 1;%max (1, ceil(100/length(subjects_list))); % number of iterations per participants

% number of starting points
niter = 2;%20;20 for paper; 2 for faster debugging
options = optimoptions('fmincon','Display','off');

%%
best_AIC = zeros(length(ms),length(ms),length(subjects_list)*nsim);
g_i=0;
for gen_m = ms
    % loop over data generating models
    g_i = g_i+1;
    gen_params = All_Params{gen_m};
    mnames{g_i} = Ms{gen_m}.ID;
    % loop over participants
    si = 0;
    k=0;
    for s = subjects_list%([4])
        k=k+1;
        X = expe_data(expe_data(:,1)==s,:);
        for sim = 1: nsim
            si = si+1;
            % simulate data from generating model, participant parameters
            % and experiment set up
            sim_data = RLWM(gen_params(k,:),X,Ms{gen_m});
            
            % fit with all models
            fk=0;
            for fit_m = ms
                fk=fk+1;
                [g_i fk si length(subjects_list)*nsim]
                fit_model = Ms{fit_m};
                pmin = fit_model.pMin;
                pmax = fit_model.pMax;
                %% actual fitting
                data = sim_data(:,[5,10,12,3,2,14]);
                % stim, action, reward, set size, block,rt
        
                sofar= [];
                j=0;
                pars  = repmat(pmin,niter,1)+rand(niter,length(pmin)).*repmat(pmax-pmin,niter,1);
                for K=2:5
                eval(['myfitfun = @(p) ',fit_model.name,'_llh(p,K,data,fit_model);'])
                    for it = 1:niter
                        par = pars(it,:);
                        j=j+1;
                        [p,fval,exitflag,output,lambda,grad,hessian] = ...
                                    fmincon(myfitfun,par,[],[],[],[],pmin,pmax,[],options);

                            sofar(j,:)=[p,K,fval];
                    end

                end
                [llh,i]=min(sofar(:,end));
                param = sofar(i(1),1:end-1);
                ntrials = size(data,1);

                % add one for capacity
                AIC = 2*llh + 2*(length(param)+1);
                BIC = 2*llh + log(ntrials)*(length(param)+1);
                AIC0 = -2*log(1/3)*ntrials;
                psr2 = (AIC0-AIC)/AIC0;

                fitmeasures(g_i,si,fk,:) = [-llh AIC BIC psr2 AIC0];
                fitparams{g_i}{fk}(si,:) = param;
                genparams{g_i}{fk}(si,:) = gen_params(k,:);
                
            end
            % figure out winning model and store information
            [~,bA] = min(squeeze(fitmeasures(g_i,si,:,2)));
            [~,bB] = min(squeeze(fitmeasures(g_i,si,:,3)));
            best_AIC(g_i,bA,si) = 1;
            best_BIC(g_i,bB,si) = 1;
        end
    end
end
%UNCOMMENT TO SAVE
save(['GenRec/GR_DS',num2str(Dataset),'M',num2str(ms)])
end

%% plot confusion matrix (Fig. S8)
% this cell generates supplementary figure 8

clear all

Datasets = [1 2 3 5 12 16];
dnames = {'CF12','SZ','EEG','fMRI','Dev','GL'};
ms = [8 26];%6 - 26; 26-38
%ms = [8 38];

figurePosition = [100, 100, 1200, 800];  % [left, bottom, width, height]
f1=figure('position',figurePosition);
for dk=1:6
    Dataset = Datasets(dk);
load(['GenRec/GR_DS',num2str(Dataset),'M',num2str(ms)],'best_AIC','mnames')
X=mean(best_AIC,3);
%mean(best_BIC,3)
% Create a colormap visualization of the matrix
subplot(2,3,dk)
imagesc(X,[0,1]);
colorbar; % Adds a colorbar to the figure
colormap('copper'); % Sets the colormap to 'jet' (you can change this as needed)

% Annotating each cell with the corresponding matrix value
[rows, cols] = size(X);
for i = 1:rows
    for j = 1:cols
        text(j, i, num2str(X(i,j)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'w');
    end
end

% Optional: set the axis properties for better visualization
axis equal tight;  % Makes sure the spacing is equal and fits tightly
title(dnames{dk});
xlabel('Fit model');
ylabel('Gen model');
set(gca,'xtick',1:length(mnames),'ytick',1:length(mnames),'xticklabels',mnames,'fontsize',14)
xtickangle(45)
end

filename = ['Figures/ConfusionMatrixAIC_Ms',num2str(num2str(ms))];
%UNCOMMENT TO SAVE
%savefig(f1,[filename,'.fig'])
%print([filename,'.png'], '-dpng');
%% plot param recovery (Fig. S9)
% this cell generates supplementary figure 9
clear all

Datasets = [1 2 3 5 12 16];
dnames = {'CF12','SZ','EEG','fMRI','Dev','GL'};
ms = [8 26];%[6 26];%6 - 26; 26-38

for m = 2%1:length(ms) 
    figurePosition = [100, 100, 1600, 1200];  % [left, bottom, width, height]

    % Create the figure with the specified size and position
    f1=figure('Position', figurePosition);
    for dk=1:6
        Dataset = Datasets(dk);
        load(['GenRec/GR_DS',num2str(Dataset),'M',num2str(ms)],...
            'genparams','fitparams','Ms')
        genP = genparams{m}{m};
        fitP = fitparams{m}{m};
        [rho,pval]=corr(genP,fitP,'type','spearman');
        rhos(dk,:) = diag(rho);
        pvals(dk,:)=diag(pval);
        fitP(:,end) = fitP(:,end) + .05*randn(size(fitP,1),1);
        for p = 1:size(genP,2)
            subplot(6,size(genP,2),p +(dk-1)*size(genP,2))
            hold on
            plot(genP(:,p),fitP(:,p),'.')
            lsline
            plot(xlim,xlim,'k--')
            title([Ms{ms(m)}.pnames{find(Ms{ms(m)}.thetamapping==p,1)},'-',dnames{dk}])
            xlabel('true')
            ylabel('fit')
        end
    end
    filename = ['Figures/CGenRec_M',Ms{ms(m)}.ID,'-Ms',num2str(num2str(ms))];
%     savefig(f1,[filename,'.fig'])
%     print([filename,'.png'], '-dpng');
end