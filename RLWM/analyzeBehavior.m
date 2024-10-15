% Anne Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This function performs behavioral analysis on a real or simulated data
% set, and returns summary statistics to be plotted at the group level.


function [LC, betas, errors,erLC] = analyzeBehavior(X)
% perform behavioral analysis on formatted individual participant data set
% input: matrix X of individual participant data
% output: summary statistics of behavior
% LC: (max Set size * number of iterations) learning curves per set size 
% beta: logistic regression coefficients
% errors: (max set size * 4) chosen/unchosen error; mean difference; mean sign 
% erLC: (max set zie *2) mean difference for early vs. late trials

% consider only trials with RT>150ms
T = find(X(:,14)>0.15);

% extract relevant info
block = X(T,2);% block number
ns = X(T,3);% set size
iter = X(T,8);% iteration nubmer
cor = X(T,12);% choice accuracy
pcor = X(T,16);% number of previous correct trials/stim
pinc = iter - pcor - 1;% number of previous incorrect trials
delay = X(T,17);% number of trials since last correct

% get unique set sizes and iterations per stimuli for the data set
setsizes = unique(ns');
iters = unique(iter');

% get learning curves
for n = setsizes(setsizes>1)
    for i = 1:10
        T = find(iter==i & ns==n);
        LC(n,i) = mean(cor(T));
    end    
end

% no logistic regression analysis here
betas = nan;
% error analysis
[errors,erLC] = erroranalyzis(X);
end


function [errors,erLC] = erroranalyzis(X)
%% compute number of past errors for each stimulus and action at each time
% point
X (:,18)=nan;
bl = unique(X(:,2)');
% for each block
for b = bl
    % block specific data
    Tb = find(X(:,2)==b);
    Xb = X(Tb,:);
    % initialize of counting of specific number of same errors
    Prev = nan*Xb(:,1);
    ns = max(Xb(:,5));
    for s = 1:ns
        for a=1:3
            % find the index of the when a stimulus and action were chosen
            T = find(Xb(:,5)==s & Xb(:,10)==a);
            % same the number of times it was previously chosen (which is
            % the length of the vector so far).
            Prev(T) = 0:length(T)-1;
        end
    end
    % store the number of past errors for the chosen error in column 18
    X(Tb,18) = Prev;
end

%% aggregate
% loop over set size
for ns = unique(X(:,3)')
    % find error trials with at least one previous stimulus exposure
    Terr = find(X(:,12)==0 & X(:,3)==ns & X(:,8)>1);
    %     Terr = find(X(:,12)==0 & X(:,3)==ns & X(:,8)>1 & X(:,8)<6);
    %     Terr = find(X(:,12)==0 & X(:,3)==ns & X(:,8)>5 & X(:,8)<11);
    % what the iteration number was
    iter = X(Terr,8);
    % number of previous correct choices
    pcor = X(Terr,16);
    % number of times this action was previously chosen
    pchos = X(Terr,18);
    % number of times the unchosen error was previously chosen.
    % on error trials, pcor + pchos + punch = iter-1.
    punch = iter - pcor - pchos - 1;
    % consider trials where the difference is not 0
    T = 1:length(pchos);
    if isempty(T)
        numerr = 0;
    else
        numerr=T(end);
    end
    errors(ns,:) = [mean(pchos) mean(punch) mean(punch(T)-pchos(T)) mean(sign(punch(T)>pchos(T))) numerr];
    
    % early trials error repeat vs. avoid
    T = find(X(Terr,8)<6);
    if isempty(T)
        erLC(ns,1)=nan;
    else
        erLC(ns,1)=mean(punch(T)-pchos(T));
    end
    
    % late trials
    T = find(X(Terr,8)>5);
    if isempty(T)
        erLC(ns,2)=nan;
    else
        erLC(ns,2)=mean(punch(T)-pchos(T));
    end
end


end