% Anne GE Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This function computes the log-likelihood llh of the data under parameters
% (theta,K) for model M.

function llh = RLWM_llh(theta,K,data,M)
% This function computes the likelihood of the model
% theta: vector of parameters (size depends on model M)
% K: discrete capacity value
% data: data matrix
% M: structure containing information about model to be simulated.

%% define model parameters  (see RLWM.m

beta = 25;
alpha = theta(M.thetamapping(1));%.2;
stick = theta(M.thetamapping(2));
rho = theta(M.thetamapping(3));
forget = theta(M.thetamapping(4));
epsilon = theta(M.thetamapping(5));%

if M.pfixed(6)
    biasWM = M.fixedvalue(6);
else
    biasWM = theta(M.thetamapping(6));
end

if M.pfixed(7)
    biasRL = M.fixedvalue(7);
else
    biasRL = theta(M.thetamapping(7));
end
if M.pfixed(8)
    r0 = M.fixedvalue(8);
else
    r0 = theta(M.thetamapping(8));
end
if length(M.pfixed)<10
    chunk = 0;
elseif M.pfixed(9)
    chunk = M.fixedvalue(9);
else
    chunk = theta(M.thetamapping(9));
end

alphaRL = [alpha*biasRL alpha];
alphaWM = [biasWM 1];

if M.value==1
    or = [r0,1];
else
    or = [1,1];
end

%% extract relevant data

Allstimuli = data(:,1);
Allchoices = data(:,2);
Allrewards = data(:,3);
Allsetsize = data(:,4);
Allblocks = data(:,5);
Allrts = data(:,6);


%% run through the data
blocks = unique(Allblocks)';
nA = 3;
llh = 0;% log likelihood initialization
for bl = blocks
    % extract relevant trials for this block
    Tb = find(Allblocks == bl & Allrts>.15);
    stimuli = Allstimuli(Tb);
    choices = Allchoices(Tb);
    rewards = Allrewards(Tb);
    ns = Allsetsize(Tb(1));
    
    % WM weight
    w = rho*min(1,K/ns);
    if M.interact
        wint=w;
    else
        wint = 0;
    end
    
    % initialize
    Q = (1/nA)*ones(ns,nA);%[.5 .5;.5 .5];
    WM = (1/nA)*ones(ns,nA);%[.5 .5;.5 .5];
    lt = log(1/nA);
    llh = llh + lt;
    
    for k = 2:length(choices)
        % trial's stimulus, choice, and outcome
        s = stimuli(k-1);
        choice = choices(k-1);
        r = rewards(k-1);
        
        % update the model
        WM = WM + forget*(1/nA - WM);
        rpe = or(r+1)-(wint*WM(s,choice) + (1-wint)*Q(s,choice));
        Q(s,choice) = Q(s,choice) + alphaRL(r+1)*rpe;
        WM(s,choice) = WM(s,choice) + alphaWM(r+1)*(r-WM(s,choice));
        % sticky
        side=zeros(1,nA);
        side(choice)=1;
        % policy compression
        if chunk>0
            for s=1:ns
                W = Q(stimuli(k),:)+stick*side;
                bRLs(s,:) = exp(beta*W);
                bRLs(s,:) = epsilon/nA + (1-epsilon)*bRLs(s,:)/sum(bRLs(s,:));
                W = WM(stimuli(k),:)+stick*side;
                bWMs(s,:) = exp(beta*W);
                bWMs(s,:) = epsilon/nA + (1-epsilon)*bWMs(s,:)/sum(bWMs(s,:));
            end
            bs = w*mean(bWMs) + (1-w)*mean(bRLs);
        else
            bs=0;
        end
        % overall policy
        W = Q(stimuli(k),:)+stick*side+chunk*bs;
        bRL = exp(beta*W);
        bRL = epsilon/nA + (1-epsilon)*bRL/sum(bRL);
        W = WM(stimuli(k),:)+stick*side+chunk*bs;
        bWM = exp(beta*W);
        bWM = epsilon/nA + (1-epsilon)*bWM/sum(bWM);
        b = w*bWM + (1-w)*bRL;
        % log-likelihood of next trial
        lt = log(b(choices(k)));
        % increment
        llh = llh + lt;
        ls(k)=lt;
    end
end
% return -llh for minimization
llh = -llh;
end