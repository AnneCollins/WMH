% Anne Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This script performs model and parameter identification (Figs. S13 and 14).

clear all
tic
% turn to true to re-run . leave false to just plot the output. 
regenerateGenRec = false;
% turn true to run a relatively fast demo of gen rec. turn false to run the
% full gen rec fitting (SLOW).
Demo = true;
%% define data set and models to consider
Datasets = [14];
dk=1;
Dataset = Datasets(dk);
dnames = {'McD'};
dname = dnames{dk};
ms = [1 7 16];
ms = [6 7 16];

%% load data and subjects and fit params

load(['Fits/FitRL2_dataset14'])

load(['DataSets/Expe',num2str(Dataset)])%

subjects_list = unique(expe_data(:,1)');
nsim = max (1, ceil(100/length(subjects_list))); % number of iterations per participants

options = optimoptions('fmincon','Display','off');

niter=10;
if regenerateGenRec
    if Demo
        % parameters for faster run
        niter = 2;
        nsim=2;
        subjects_list=subjects_list(1:2);
    end
    
    %%
    best_AIC = zeros(length(ms),length(ms),length(subjects_list)*nsim);
    g_i=0;
    for gen_m = ms
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
                sim_data = RL2(gen_params(k,:),X,Ms{gen_m});
                
                % fit with all models
                fk=0;
                for fit_m = ms
                    fk=fk+1;
                    [g_i fk si length(subjects_list)*nsim]
                    fit_model = Ms{fit_m};
                    pmin = fit_model.pMin;
                    pmax = fit_model.pMax;
                    %% actual fitting
                    data = sim_data(:,[5,10,13,3,2,14]);
                    % stim, action, reward, set size, block,rt
                    %%
                    sofar= [];
                    j=0;
                    pars  = repmat(pmin,niter,1)+rand(niter,length(pmin)).*repmat(pmax-pmin,niter,1);
                    eval(['myfitfun = @(p) ',fit_model.name,'_llh(p,data,fit_model);'])
                    for it = 1:niter
                        par = pars(it,:);
                        j=j+1;
                        [p,fval,exitflag,output,lambda,grad,hessian] = ...
                            fmincon(myfitfun,par,[],[],[],[],pmin,pmax,[],options);
                        
                        sofar(j,:)=[p,fval];
                    end
                    
                    [llh,i]=min(sofar(:,end));
                    param = sofar(i(1),1:end-1);
                    ntrials = size(data,1);
                    %%
                    AIC = 2*llh + 2*(length(param));
                    BIC = 2*llh + log(ntrials)*(length(param));
                    AIC0 = -2*log(1/3)*ntrials;
                    psr2 = (AIC0-AIC)/AIC0;
                    
                    fitmeasures(g_i,si,fk,:) = [-llh AIC BIC psr2 AIC0];
                    fitparams{g_i}{fk}(si,:) = param;
                    genparams{g_i}{fk}(si,:) = gen_params(k,:);
                    
                end
                [~,bA] = min(squeeze(fitmeasures(g_i,si,:,2)));
                [~,bB] = min(squeeze(fitmeasures(g_i,si,:,3)));
                best_AIC(g_i,bA,si) = 1;
                best_BIC(g_i,bB,si) = 1;
            end
        end
    end
    
    save(['GenRec/GR_DS',num2str(Dataset),'M',num2str(ms)])
end
%% plot confusion matrix
%
clear all
%
ms = [6 7 16];
Datasets = [14];
figurePosition = [100, 100, 400, 400];  % [left, bottom, width, height]
f1=figure('position',figurePosition);
for dk=1
    Dataset = Datasets(dk);
    load(['GenRec/GR_DS',num2str(Dataset),'M',num2str(ms)],'best_AIC','mnames')
    X=mean(best_AIC,3);
    %mean(best_BIC,3)
    % Create a colormap visualization of the matrix
    %subplot(2,3,dk)
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
    title('RLWMP');
    xlabel('Fit model');
    ylabel('Gen model');
    set(gca,'xtick',1:length(mnames),'ytick',1:length(mnames),'xticklabels',mnames,'fontsize',14)
    xtickangle(45)
end

filename = ['Figures/ConfusionMatrixAIC-Ms',num2str(num2str(ms))];
% UNCOMMENT TO SAVE
% savefig(f1,[filename,'.fig'])
% print([filename,'.png'], '-dpng');
% print([filename,'.svg'], '-dsvg');
%% plot param recovery
clear all
ms = [1 7 16];
Datasets = [14];
dnames = {'McD'};
%
for m = 2%1:length(ms)
    figurePosition = [100, 100, 600, 600];  % [left, bottom, width, height]
    
    % Create the figure with the specified size and position
    f1=figure('Position', figurePosition);
    for dk=1%:6
        Dataset = Datasets(dk);
        load(['GenRec/GR_DS',num2str(Dataset),'M',num2str(ms)],...
            'genparams','fitparams','Ms')
        genP = genparams{m}{m};
        fitP = fitparams{m}{m};
        [rho,pval]=corr(genP,fitP,'type','spearman');
        rhos(dk,:) = diag(rho)';
        pvals(dk,:)=diag(pval)';
        %genP(:,1) = log(genP(:,1));
        %fitP(:,1) = log(fitP(:,1));
        fitP(:,end) = fitP(:,end);
        for p = 1:size(genP,2)
            subplot(3,3,p +(dk-1)*size(genP,2))
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
    % UNCOMMENT TO SAVE
    %     savefig(f1,[filename,'.fig'])
    %     print([filename,'.png'], '-dpng');
    %     print([filename,'.svg'], '-dsvg');
end
%%

[rho, pval] = corr(fitP,fitP,'type','spearman');
toc