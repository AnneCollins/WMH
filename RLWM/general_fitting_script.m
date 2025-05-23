% Anne Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This script performs model fitting and defines the models. 
% This script is ***slow*** to run (if running all models and data sets
% without parallelization, it will take multiple days). 
% run in demo mode for a demo.

% this script generated all the Fits in the Fits/ folder.
addpath("Models/")

clear all;
%% set up a demo
Demo = true;
% runs 2 subjects on 2 models with fewer iterations
% turn to false to run the whole thing.

%%
% define data-sets
Datasets = [1 2 3 5 12 16];
% specific data set to be fitted
Dataset = Datasets(1);
WL =0;
%% load data set and subject list

load(['DataSets/Expe',num2str(Dataset)])%WML_data.mat
subjects_list = unique(expe_data(:,1)');

% already fitted some models, load previously fitted models
load(['NewFits/FitRLWM_dataset',num2str(Dataset)])
%% set params
Ms = [];

%% RLWM - classic RLWM model family

curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 1 1 1 0];
curr_model.fixedvalue = [nan nan nan nan nan 0 0 0 nan];
curr_model.thetamapping = [1 2 3 4 5 nan nan nan 6];
curr_model.pMin = [0 -1 0 0 0];
curr_model.pMax = [1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM0 RL0 0';

Ms{1}=curr_model;


curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 0 1 1 0];
curr_model.fixedvalue = [nan nan nan nan nan nan 0 0 nan];
curr_model.thetamapping = [1 2 3 4 5 6 nan nan 7];
curr_model.pMin = [0 -1 0 0 0 0];
curr_model.pMax = [1 1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM RL0 0';

Ms{2}=curr_model;



curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 0 1 1 0];
curr_model.fixedvalue = [nan nan nan nan nan nan 1 0 nan];
curr_model.thetamapping = [1 2 3 4 5 6 nan nan 7];
curr_model.pMin = [0 -1 0 0 0 0];
curr_model.pMax = [1 1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM RL1 0';

Ms{3}=curr_model;


curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 1 0 1 0];
curr_model.fixedvalue = [nan nan nan nan nan 0 nan 0 nan];
curr_model.thetamapping = [1 2 3 4 5 nan 6 nan 7];
curr_model.pMin = [0 -1 0 0 0 0];
curr_model.pMax = [1 1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM0 RL 0';

Ms{4}=curr_model;


curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 1 0 1 0];
curr_model.fixedvalue = [nan nan nan nan nan 1 nan 0 nan];
curr_model.thetamapping = [1 2 3 4 5 nan 6 nan 7];
curr_model.pMin = [0 -1 0 0 0 0];
curr_model.pMax = [1 1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM1 RL 0';

Ms{5}=curr_model;

curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 0 0 1 0];
curr_model.fixedvalue = [nan nan nan nan nan nan nan 0 nan];
curr_model.thetamapping = [1 2 3 4 5 6 6 nan 7];
curr_model.pMin = [0 -1 0 0 0 0];
curr_model.pMax = [1 1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM=RL 0';

Ms{6}=curr_model;


curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 0 0 1 0];
curr_model.fixedvalue = [nan nan nan nan nan nan nan 0 nan];
curr_model.thetamapping = [1 2 3 4 5 6 7 nan 8];
curr_model.pMin = [0 -1 0 0 0 0 1];
curr_model.pMax = [1 1 1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM RL 0';

Ms{7}=curr_model;



curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 1 1 1 0];
curr_model.fixedvalue = [nan nan nan nan nan 1 0 0 nan];
curr_model.thetamapping = [1 2 3 4 5 nan nan nan 6];
curr_model.pMin = [0 -1 0 0 0];
curr_model.pMax = [1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM1 RL0 0';

Ms{8}=curr_model;

curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 1 1 1 0];
curr_model.fixedvalue = [nan nan nan nan nan 1 1 0 nan];
curr_model.thetamapping = [1 2 3 4 5 nan nan nan 6];
curr_model.pMin = [0 -1 0 0 0];
curr_model.pMax = [1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM1 RL1 0';

Ms{9}=curr_model;

curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 1 1 1 0];
curr_model.fixedvalue = [nan nan nan nan nan 0 1 0 nan];
curr_model.thetamapping = [1 2 3 4 5 nan nan nan 6];
curr_model.pMin = [0 -1 0 0 0];
curr_model.pMax = [1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM0 RL1 0';

Ms{10}=curr_model;


%% 11-10: RLWM - int add RL-WM interaction. Not explored in this paper

for i = [3:7 9:10]
    curr_model=Ms{i};
    curr_model.interact = 1;
    curr_model.ID = ['int ',curr_model.ID];

    Ms{10+i}=curr_model;
end

%% 21-30 WMH
for i = [3:7 9:10]
    curr_model=Ms{i};
    curr_model.value = 0;
    curr_model.ID = ['freq ',curr_model.ID];

    Ms{i+20}=curr_model;
end
%% 31-40 neg outcome RL weight - commented out models are redundant
% 
% curr_model = [];
% curr_model.name = 'RLWM';
% curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','0','K'};
% curr_model.pfixed = [0 0 0 0 0 1 1 1 0];
% curr_model.fixedvalue = [nan nan nan nan nan 0 1 0 nan];
% curr_model.thetamapping = [1 2 3 4 5 nan nan nan 6];
% curr_model.pMin = [0 -1 0 0 0];
% curr_model.pMax = [1 1 1 1 1];
% curr_model.interact = 0;
% curr_model.value = 1;
% 
% Ms{31}=curr_model;

curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 1 1 0 0];
curr_model.fixedvalue = [nan nan nan nan nan 0 1 nan nan];
curr_model.thetamapping = [1 2 3 4 5 nan nan 6 7];
curr_model.pMin = [0 -1 0 0 0 0];
curr_model.pMax = [1 1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM0 RL1 r0';

Ms{32}=curr_model;

% curr_model = [];
% curr_model.name = 'RLWM';
% curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','0','K'};
% curr_model.pfixed = [0 0 0 0 0 1 1 1 0];
% curr_model.fixedvalue = [nan nan nan nan nan 0 1 1 nan];
% curr_model.thetamapping = [1 2 3 4 5 nan nan nan 6];
% curr_model.pMin = [0 -1 0 0 0];
% curr_model.pMax = [1 1 1 1 1];
% curr_model.interact = 0;
% curr_model.value = 1;
% 
% Ms{33}=curr_model;

% 
% curr_model = [];
% curr_model.name = 'RLWM';
% curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','0','K'};
% curr_model.pfixed = [0 0 0 0 0 0 1 1 0];
% curr_model.fixedvalue = [nan nan nan nan nan nan 1 0 nan];
% curr_model.thetamapping = [1 2 3 4 5 6 nan nan 7];
% curr_model.pMin = [0 -1 0 0 0 0 0];
% curr_model.pMax = [1 1 1 1 1 1 1];
% curr_model.interact = 0;
% curr_model.value = 1;
% 
% Ms{34}=curr_model;

curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 0 1 0 0];
curr_model.fixedvalue = [nan nan nan nan nan nan 1 nan nan];
curr_model.thetamapping = [1 2 3 4 5 6 nan 7 8];
curr_model.pMin = [0 -1 0 0 0 0 0 0];
curr_model.pMax = [1 1 1 1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM RL1 r0';

Ms{35}=curr_model;

% curr_model = [];
% curr_model.name = 'RLWM';
% curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','0','K'};
% curr_model.pfixed = [0 0 0 0 0 0 1 1 0];
% curr_model.fixedvalue = [nan nan nan nan nan 1 1 1 nan];
% curr_model.thetamapping = [1 2 3 4 5 6 nan nan 7];
% curr_model.pMin = [0 -1 0 0 0 0];
% curr_model.pMax = [1 1 1 1 1 1];
% curr_model.interact = 0;
% curr_model.value = 1;
% 
% Ms{36}=curr_model;
% 
% curr_model = [];
% curr_model.name = 'RLWM';
% curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','0','K'};
% curr_model.pfixed = [0 0 0 0 0 1 1 1 0];
% curr_model.fixedvalue = [nan nan nan nan nan 1 1 0 nan];
% curr_model.thetamapping = [1 2 3 4 5 nan nan nan 6];
% curr_model.pMin = [0 -1 0 0 0];
% curr_model.pMax = [1 1 1 1 1];
% curr_model.interact = 0;
% curr_model.value = 1;
% 
% Ms{37}=curr_model;

curr_model = [];
curr_model.name = 'RLWM';
curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','r0','K'};
curr_model.pfixed = [0 0 0 0 0 1 1 0 0];
curr_model.fixedvalue = [nan nan nan nan nan 1 1 nan nan];
curr_model.thetamapping = [1 2 3 4 5 nan nan 6 7];
curr_model.pMin = [0 -1 0 0 0 0];
curr_model.pMax = [1 1 1 1 1 1];
curr_model.interact = 0;
curr_model.value = 1;
curr_model.ID = 'WM1 RL1 r0';

Ms{38}=curr_model;
% 
% curr_model = [];
% curr_model.name = 'RLWM';
% curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','0','K'};
% curr_model.pfixed = [0 0 0 0 0 1 1 1 0];
% curr_model.fixedvalue = [nan nan nan nan nan 1 1 1 nan];
% curr_model.thetamapping = [1 2 3 4 5 nan nan nan 6];
% curr_model.pMin = [0 -1 0 0 0];
% curr_model.pMax = [1 1 1 1 1];
% curr_model.interact = 0;
% curr_model.value = 1;
% 
% Ms{39}=curr_model;

%% 41-50: chunking (policy compression free param)

for i = 1:10
    
    curr_model=Ms{i};
    curr_model.name = 'RLWM';
    curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM',...
        'biasRL','r0','chunk','K'};
    curr_model.pfixed = [curr_model.pfixed 0];
    curr_model.fixedvalue = [curr_model.fixedvalue nan];
    curr_model.thetamapping = [curr_model.thetamapping 2+length(curr_model.pMin)];
    curr_model.pMin = [curr_model.pMin 0];
    curr_model.pMax = [curr_model.pMax 1];
    curr_model.ID = [curr_model.ID,'-PC'];
    Ms{40+i}=curr_model;
end


%% 51-60. WM learning rate. best is 52, worse than best RLW (1-10)

for i = 1:10
    
    curr_model=Ms{i};
    curr_model.name = 'RLWM_R1';
    curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM',...
        'biasRL','r0','chunk','alphaWM1','forgetRL','K'};
    curr_model.pfixed = [curr_model.pfixed(1:8) 1 0 1 0];
    curr_model.fixedvalue = [curr_model.fixedvalue(1:8) 0 nan 0];
    curr_model.pMin = [curr_model.pMin 0];
    curr_model.pMax = [curr_model.pMax 1]; 
    curr_model.thetamapping = [curr_model.thetamapping(1:end-1) nan length(curr_model.pMin) nan 1+length(curr_model.pMin)];
    curr_model.ID = [curr_model.ID,'-WMalpha'];
    Ms{50+i}=curr_model;
end

%% 61-70: RL forgetting (61-70)

for i = 1:10
    
    curr_model=Ms{i};
    curr_model.name = 'RLWM_R1';
    curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM',...
        'biasRL','r0','chunk','alphaWM1','forgetRL','K'};
    curr_model.pfixed = [curr_model.pfixed(1:8) 1 1 0];
    curr_model.fixedvalue = [curr_model.fixedvalue(1:8) 0 0 nan];
    curr_model.pMin = [curr_model.pMin 0];
    curr_model.pMax = [curr_model.pMax 1]; 
    curr_model.thetamapping = [curr_model.thetamapping(1:end-1) nan length(curr_model.pMin) nan 1+length(curr_model.pMin)];
    curr_model.ID = [curr_model.ID,'-RLforget'];
    Ms{60+i}=curr_model;
end
%boum

%% 71:80 - RL-H
for i = [3:7 9:10]
    
    curr_model=Ms{i};
    curr_model.name = 'RLWM_R1';
    curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM',...
        'biasRL','r0','chunk','alphaWM1','forgetRL','K'};
    curr_model.pfixed = [curr_model.pfixed(1:7) 1 1 0 1];
    curr_model.fixedvalue = [curr_model.fixedvalue(1:7) 1 0 nan 0];
    curr_model.pMin = [curr_model.pMin 0];
    curr_model.pMax = [curr_model.pMax 1]; 
    curr_model.thetamapping = [curr_model.thetamapping(1:end-1) nan length(curr_model.pMin) nan 1+length(curr_model.pMin)];
    curr_model.ID = [curr_model.ID,'-RLH'];
    Ms{70+i}=curr_model;
end

%% 81+ : WMRLH

% shared biases
i=81;
    curr_model=[];
    curr_model.name = 'RLWMH';
    curr_model.pnames = {'alphaRL','stick','rhoWM','forget','epsilon','biasWM',...
        'biasRL','r0','chunk','rhoH','alphaH','biasH','K'};
    curr_model.pfixed = [0 0 0 0 0 0 0 1 1 0 0 0 0];
    curr_model.fixedvalue = [nan(1,7) 1 0 nan(1,4)];% fix r0 at 1 for H, chunk at 0 for no chunking
    curr_model.pMin = [0 -1 zeros(1,6)];
    curr_model.pMax = ones(1,8); 
    curr_model.thetamapping = [1:6 6 nan nan 7 8 6 nan];% shared bias parameters
    curr_model.interact = 0;
    curr_model.value = 1;
    curr_model.ID = 'RLWMH-shared bias';
    Ms{i}=curr_model;

% no WM bias, no H bias, free RL bias
i=82;
    curr_model=Ms{81};
    curr_model.pnames = {'alphaRL','stick','rhoWM','forget','epsilon','biasWM',...
        'biasRL','r0','chunk','rhoH','alphaH','biasH','K'};
    curr_model.pfixed([6 12]) = 1;% fix WM and H bias
    curr_model.fixedvalue([6 12]) = 1;% ffix them at 1
    curr_model.thetamapping = [1:5 nan 6 nan nan 7 8 nan nan];% shared bias parameters
    curr_model.ID = 'RLWMH-freeRL bias';
    Ms{i}=curr_model;
    
%% 83: RLf+H
% shared biases
i=83;
    curr_model=[];
    curr_model.name = 'RLWMH';
    curr_model.pnames = {'alphaRL','stick','rhoWM','forget','epsilon','biasWM',...
        'biasRL','r0','chunk','rhoH','alphaH','biasH','forgetRL','K'};
    curr_model.pfixed = [0 0 1 1 0 1 0 1 1 0 0 1 0];
    curr_model.fixedvalue = nan(1,13);
    curr_model.fixedvalue([3:4 6]) = [0 0 0];% no WM
    curr_model.fixedvalue([8:9]) = [1 0];% r0=1 for H, no chunk
    curr_model.fixedvalue([12]) = [1];% no H bias
    curr_model.pMin = [0 -1 zeros(1,5)];
    curr_model.pMax = ones(1,7); 
    curr_model.thetamapping = [1 2 nan nan 3 nan 4 nan nan 5 6 nan 7];%
    curr_model.interact = 0;
    curr_model.value = 1;
    curr_model.ID = 'RLfH';
    Ms{i}=curr_model;    

    
%% 91-100 RLWM + bias. 

for i = 1:10
    
    curr_model=Ms{i};
    curr_model.name = 'RLWM_R2';
    curr_model.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM',...
        'biasRL','A1','A2-A3','K'};
    curr_model.pfixed = [curr_model.pfixed(1:7) 0 0 0];
    curr_model.fixedvalue = [curr_model.fixedvalue(1:7) nan nan nan];
    curr_model.pMin = [curr_model.pMin(1:end) 0 0];
    curr_model.pMax = [curr_model.pMax(1:end) 1 1]; 
    curr_model.thetamapping = [curr_model.thetamapping(1:end-2) length(curr_model.pMin)-1 length(curr_model.pMin) nan];
    curr_model.ID = [curr_model.ID,'-bias'];
    Ms{90+i}=curr_model;
end
    
    
%% 91-100 RLWM + with no WM over capacity. 

for i = 1:10
    
    curr_model=Ms{i};
    curr_model.name = 'RLWM_R3';
    curr_model.ID = [curr_model.ID,'- WMdrop'];
    Ms{100+i}=curr_model;
end
%boum
%% models to consider for fitting
%toconsider = [1:10 10+[3:7 9:10] 20+[3:7 9:10] 32 35 38];
toconsider = [1:10 20+[3:7 9:10] 32 35 38];
%toconsider = toconsider(3:end)
toconsider = [toconsider 41:48 51:60 61:70 70+[3:7 9:10]];
toconsider = [101:110];
%% set up fitting options
options = optimoptions('fmincon','Display','off');
% set up number of starting points for optimizer
niter = 10;

%% set up demo mode

if Demo 
    toconsider = toconsider(1:2);
    niter =2;
    subjects_list = subjects_list(1:2);
end
    
%% fitting

for m=toconsider%(toconsider>70)
    %%
    fit_model = Ms{m};
    pmin = fit_model.pMin;
    pmax = fit_model.pMax;

    k=0;
    fitmeasures = [];
    fitparams = [];
    %%
    for s = subjects_list%([4])
        %%
        k=find(subjects_list==s);
        disp('Model#, Sub#, out of N subjects')
        [m k length(subjects_list)]
        %tic;
        X = expe_data(expe_data(:,1)==s,[5,10,12,3,2,14]);
        % stim, action, reward, set size, block,rt
        data = X;
        %% this data fitting
        sofar= [];
        j=0;
        % define random starting points to be used for each capacity value
        pars  = repmat(pmin,niter,1)+rand(niter,length(pmin)).*repmat(pmax-pmin,niter,1);
        % loop over capacity values
        for K=2:5
            % define model to be fit
            eval(['myfitfun = @(p) ',fit_model.name,'_llh(p,K,data,fit_model);'])
            % iterate over starting points
            for it = 1:niter
                par = pars(it,:);
                j=j+1;
                [p,fval,exitflag,output,lambda,grad,hessian] = ...
                            fmincon(myfitfun,par,[],[],[],[],pmin,pmax,[],options);
                
                        %store best likelihood, param, K
                sofar(j,:)=[p,K,fval];
            end

        end
        %% store information

        % global minimum
        [llh,i]=min(sofar(:,end));
        param = sofar(i(1),1:end-1);
        % uncomment for debugging
%        [s param llh]
%        [m k length(subjects_list) toc]
        ntrials = size(data,1);

        % compute AIC, BIC, 
        % add one for capacity
        AIC = 2*llh + 2*(length(param)+1);
        BIC = 2*llh + log(ntrials)*(length(param)+1);
        AIC0 = -2*log(1/3)*ntrials;
        psr2 = (AIC0-AIC)/AIC0;

        % store fit measures and best fit params
        fitmeasures(k,:) = [s -llh AIC BIC psr2 AIC0];
        fitparams(k,:) = param;
    end

    % store for the models in a new folder to not overwrite my fits.
    All_Params{m} = fitparams;
    All_fits(:,:,m) = fitmeasures;
    save(['NewFits/FitRLWM_dataset',num2str(Dataset)],'Ms','All_Params','All_fits','subjects_list') 
end

