
% Anne GE Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This function computes the log-likelihood llh of the data under parameters
% (theta,K) for model M.
% this is for models 91-100, which add new mechanisms. 

function llh = RLWM_R2_llh(theta,K,data,M)


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

actionbias1 = theta(M.thetamapping(8));
actionbias2 = theta(M.thetamapping(9));

policyPref = [actionbias1 (1-actionbias1)*actionbias2 (1-actionbias1)*(1-actionbias2)];

alphaRL = [alpha*biasRL alpha];
alphaWM = [biasWM*alphaWM1 1*alphaWM1];

Allstimuli = data(:,1);
Allchoices = data(:,2);
Allrewards = data(:,3);
Allsetsize = data(:,4);
Allblocks = data(:,5);
Allrts = data(:,6);

blocks = unique(Allblocks)';

llh = 0;
if M.value==1
    or = [r0,1];
else
    or = [1,1];
end

for bl = blocks
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
    
Q = (1/nA)*ones(ns,nA);%[.5 .5;.5 .5];
WM = (1/nA)*ones(ns,nA);%[.5 .5;.5 .5];
lt = log(1/nA);
llh = llh + lt;

for k = 2:length(choices)
    s = stimuli(k-1);
    choice = choices(k-1);
    r = rewards(k-1);
    
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
    lt = log(b(choices(k)));
    llh = llh + lt;
    ls(k)=lt;
end
end
llh = -llh;
end
