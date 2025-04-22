% Anne GE Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This function performs one simulation of the model.
% this is for models101-above, which add new mechanisms. 

function sim_data = RLWM_R3(theta,expe_data,M)
% 
% 
% if nargin<4
% plotting=0;
% end
K = theta(end);

nA = 3;
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

r0=0;
alphaWM1=1;
forgetRL = 0;

% actionbias1 = theta(M.thetamapping(8));
% actionbias2 = theta(M.thetamapping(9));

policyPref = 1/nA;%[actionbias1 (1-actionbias1)*actionbias2 (1-actionbias1)*(1-actionbias2)];

alphaRL = [alpha*biasRL alpha];
alphaWM = [biasWM*alphaWM1 1*alphaWM1];


Allstimuli = expe_data(:,5);
Allchoices = expe_data(:,10);
Allrewards = expe_data(:,12);
Allsetsize = expe_data(:,3);
Allblocks = expe_data(:,2);
Allrts = expe_data(:,14);
AllaCor = expe_data(:,9);
Alliter = expe_data(:,8);
Alldelay = expe_data(:,17);
Allpcor = expe_data(:,16);

sim_data = expe_data;

blocks = unique(Allblocks)';

if M.value==1
    or = [r0,1];
else
    or = [1,1];
end
for bl = blocks
    Tb = find(Allblocks == bl & Allrts>.15);
    stimuli = Allstimuli(Tb);
    choices = Allchoices(Tb);
    acors = AllaCor(Tb);
    iters = Alliter(Tb);
    rewards = Allrewards(Tb);
    ns = Allsetsize(Tb(1));

    % WM weight
    %w = rho*min(1,K/ns);
    w = rho*((K/ns)>=1);
    if M.interact
        wint=w;
    else
        wint = 0;
    end
    
    Q = (1/nA)*ones(ns,nA);
    WM = (1/nA)*ones(ns,nA);
    b= Q(1,:);
    for k = 2:length(choices)
        s = stimuli(k-1);
        if iters(k-1)==1
            choice = choices(k-1);
        else
            choice = select(b);
            choices(k-1) = choice;
        end
        r = choice == acors(k-1);
        rewards(k-1) = r;

        Q = Q + forgetRL*(1/nA - Q);
        WM = WM + forget*(1/nA - WM);
        rpe = or(r+1)-(wint*WM(s,choice) + (1-wint)*Q(s,choice));
        Q(s,choice) = Q(s,choice) + alphaRL(r+1)*rpe;
        WM(s,choice) = WM(s,choice) + alphaWM(r+1)*(r-WM(s,choice));

        side=zeros(1,nA);
        side(choice)=1;

        W = Q(stimuli(k),:)+stick*side;%+chunk*mean(Q);
        bRL = exp(beta*W);
        bRL = epsilon*policyPref + (1-epsilon)*bRL/sum(bRL);
        W = WM(stimuli(k),:)+stick*side;
        bWM = exp(beta*W);
        bWM = epsilon*policyPref + (1-epsilon)*bWM/sum(bWM);

        b = w*bWM + (1-w)*bRL;
    end
    Allrewards(Tb) = rewards;
    Allchoices(Tb) = choices;
    Alldelay(Tb) = computedelay(stimuli,rewards);
    Allpcor(Tb) = computepcor(stimuli,rewards);
    
    
    

end

sim_data(:,10) = Allchoices;
sim_data(:,12) = Allrewards;
sim_data(:,13) = Allrewards;
sim_data(:,16) = Allpcor;
sim_data(:,17) = Alldelay;
end

function a = select(pr)
a = find([0 cumsum(pr)]<rand,1,'last');
end

function delay = computedelay(stimuli,rewards);

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
pcor = 0*stimuli;
for s = unique(stimuli')
    T = find(stimuli==s);
    pcor(T) = cumsum(rewards(T))-rewards(T);
end
end