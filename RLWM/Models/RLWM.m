% Anne GE Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This function performs one simulation of the model.

function sim_data = RLWM(theta,expe_data,M)
% This function performs one simulation of the model, and returns simulated
% data in the same format as experimental data
% input
% theta: vector of parameters (size depends on model M)
% expe_data: experimental data matrix that constrains the experiment on
% which the model is simulated (e.g. sequence of blocks, stimuli, correct
% actions, etc.)
% M: structure containing information about model to be simulated.

%% define model parameters 
K = theta(end);% capacity
beta = 25;% softmax noise
alpha = theta(M.thetamapping(1));%learning rate
stick = theta(M.thetamapping(2));% motor perseveration
rho = theta(M.thetamapping(3));% WM weight
epsilon = theta(M.thetamapping(5));%lapse rate
forget = theta(M.thetamapping(4));%WM decay
% model dependent parameters
if M.pfixed(6)
    biasWM = M.fixedvalue(6);% WM learning rate bias
else
    biasWM = theta(M.thetamapping(6));
end
% RL learning rate bias
if M.pfixed(7)
    biasRL = M.fixedvalue(7);
else
    biasRL = theta(M.thetamapping(7));
end
% RL r0 value (r0=1 --> H model)
if M.pfixed(8)
    r0 = M.fixedvalue(8);
else
    r0 = theta(M.thetamapping(8));
end
% policy compression
if length(M.pfixed)<10
    chunk = 0;
elseif M.pfixed(9)
    chunk = M.fixedvalue(9);
else
    chunk = theta(M.thetamapping(9));
end
% set up learning rates [alpha- alpha+]
alphaRL = [alpha*biasRL alpha];
alphaWM = [biasWM 1];

if M.value==1
    or = [r0,1];
else
    or = [1,1];
end

%% set up the simulation experiment

% extract relevant experimental data
Allstimuli = expe_data(:,5);% sequence of stimuli
Allchoices = expe_data(:,10);% sequence of choices
Allrewards = expe_data(:,12);% sequence of rewards
Allsetsize = expe_data(:,3);% sequence of set sizes
Allblocks = expe_data(:,2);% sequence of blocks
Allrts = expe_data(:,14);% sequence of reaction times
AllaCor = expe_data(:,9);% sequence of target correct action
Alliter = expe_data(:,8);% sequence of iteration number
Alldelay = expe_data(:,17);% sequence of delay
Allpcor = expe_data(:,16);% sequence of number of previous correct

% create new simulation data
sim_data = expe_data;

% identify number of blocks 
blocks = unique(Allblocks)';

% number of actions
nA = 3;


%% start the simulation
% loop over blocks
for bl = blocks
    % simulate only for relevant trials (as in data analysis)
    Tb = find(Allblocks == bl & Allrts>.15);
    % extract experiment information for current block
    stimuli = Allstimuli(Tb);
    choices = Allchoices(Tb);
    acors = AllaCor(Tb);
    iters = Alliter(Tb);
    rewards = Allrewards(Tb);
    ns = Allsetsize(Tb(1));

    % WM weight
    w = rho*min(1,K/ns);
    if M.interact
        wint=w;
    else
        wint = 0;
    end
    
    % initialize RL/H weights, and WM weights.
    Q = (1/nA)*ones(ns,nA);
    WM = (1/nA)*ones(ns,nA);
    b= Q(1,:);
    %loop over trials - 
    for k = 2:length(choices)
        s = stimuli(k-1);
        if iters(k-1)==1
            % make first choice same as participants
            choice = choices(k-1);
        else
            % make a choice
            choice = select(b);
            % store choice
            choices(k-1) = choice;
        end
        % reward is deterministic if choice corresponds to target choice
        r = choice == acors(k-1);
        % store reward
        rewards(k-1) = r;

        % model updates
        % WM decay
        WM = WM + forget*(1/nA - WM);
        % compute RL/H RPE
        rpe = or(r+1)-(wint*WM(s,choice) + (1-wint)*Q(s,choice));
        % update RL/H
        Q(s,choice) = Q(s,choice) + alphaRL(r+1)*rpe;
        %update WM
        WM(s,choice) = WM(s,choice) + alphaWM(r+1)*(r-WM(s,choice));

        % set up sticky choice
        side=zeros(1,nA);
        side(choice)=1;
        
        % compute policy compression
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
        
        % set up RL policy with sticky and compression
        W = Q(stimuli(k),:)+stick*side+chunk*bs;
        bRL = exp(beta*W);
        % include random lapses
        bRL = epsilon/nA + (1-epsilon)*bRL/sum(bRL);
        % set up WM policy with sticky and compression
        W = WM(stimuli(k),:)+stick*side+chunk*bs;
        bWM = exp(beta*W);
        % include random lapses
        bWM = epsilon/nA + (1-epsilon)*bWM/sum(bWM);
        % overall policy is the mixture
        b = w*bWM + (1-w)*bRL;
    end
    % store rewards and choices for this block
    Allrewards(Tb) = rewards;
    Allchoices(Tb) = choices;
    % compute and store derived variables delay and pcor
    Alldelay(Tb) = computedelay(stimuli,rewards);
    Allpcor(Tb) = computepcor(stimuli,rewards);
    
    
    

end

% store simulation data
sim_data(:,10) = Allchoices;
sim_data(:,12) = Allrewards;
sim_data(:,13) = Allrewards;
sim_data(:,16) = Allpcor;
sim_data(:,17) = Alldelay;
end

function a = select(pr)
% implement random choice selection based on a probability distribution
a = find([0 cumsum(pr)]<rand,1,'last');
end

function delay = computedelay(stimuli,rewards);
% compute delay since last correct
delay = nan*stimuli;
for s=unique(stimuli')
    T = find(stimuli==s);
    for t = T(2:end)
        y = find(stimuli(1:t-1)==s&rewards(1:t-1)==1,1,'last');
        if ~isempty(y)
            delay(t)=t-y;
        end
    end
end

end

function pcor = computepcor(stimuli, rewards)
% compute pcor
pcor = 0*stimuli;
for s = unique(stimuli')
    T = find(stimuli==s);
    pcor(T) = cumsum(rewards(T))-rewards(T);
end
end