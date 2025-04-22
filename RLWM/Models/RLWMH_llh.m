% Anne GE Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This function computes the log-likelihood llh of the data under parameters
% (theta,K) for model M.
% this is for models 81-81, which consider 3 processes (RL, WM, H). 

function llh = RLWMH_llh(theta,K,data,M)
% 
% 
% if nargin<4
% plotting=0;
% end

nA = 3;
beta = 25;

% M.pnames = {'alphaRL','stick','rho','forget','epsilon','biasWM','biasRL','K'};
% M.pfixed = [0 0 0 0 0 1 1 0];
% M.fixedvalue = [nan nan nan nan nan 0 0 nan];
% M.thetamapping = [1 2 3 4 5 nan nan 6];

alphaRL = theta(M.thetamapping(1));%.2;
stick = theta(M.thetamapping(2));
if M.pfixed(3)
    rhoWM = M.fixedvalue(3);
else
rhoWM = theta(M.thetamapping(3));
end
if M.pfixed(3)
    forget = M.fixedvalue(4);
else
forget = theta(M.thetamapping(4));
end
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

% extend the RL part of the model


if length(M.pfixed)<13
    
else
    if M.pfixed(10)
        rhoH = M.fixedvalue(10);
    else
        rhoH = theta(M.thetamapping(10));
    end
    if M.pfixed(11)
        alphaH = M.fixedvalue(11);
    else
        alphaH = theta(M.thetamapping(11));
    end
    if M.pfixed(12)
        biasH = M.fixedvalue(12);
    else
        biasH = theta(M.thetamapping(12));
    end
end

if length(M.pfixed)<14
forgetRL=0; 
else 
    if M.pfixed(13)
        forgetRL = M.fixedvalue(13);
    else
        forgetRL = theta(M.thetamapping(13));
    end
end
alphaWM1 = 1;

alphaRL = [alphaRL*biasRL alphaRL];
alphaWM = [biasWM*alphaWM1 1*alphaWM1];
alphaH = [alphaH*biasH alphaH];

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
    wWM = rhoWM*min(1,K/ns);
    if M.interact
        wint=wWM;
    else
        wint = 0;
    end
    
Q = (1/nA)*ones(ns,nA);%[.5 .5;.5 .5];
H = (1/nA)*ones(ns,nA);%[.5 .5;.5 .5];
WM = (1/nA)*ones(ns,nA);%[.5 .5;.5 .5];
lt = log(1/nA);
llh = llh + lt;

for k = 2:length(choices)
    s = stimuli(k-1);
    choice = choices(k-1);
    r = rewards(k-1);

    Q = Q + forgetRL*(1/nA - Q);
    WM = WM + forget*(1/nA - WM);
    WM(s,choice) = WM(s,choice) + alphaWM(r+1)*(r-WM(s,choice));
    rpe = r-(wint*WM(s,choice) + (1-wint)*Q(s,choice));
    Q(s,choice) = Q(s,choice) + alphaRL(r+1)*rpe;
    rpe = or(r+1)-(wint*WM(s,choice) + (1-wint)*Q(s,choice));
    H(s,choice) = H(s,choice) + alphaH(r+1)*rpe;
     
    side=zeros(1,nA);
    side(choice)=1;
      
        %VBRL policy
        W = Q(stimuli(k),:)+stick*side;
        bRL = exp(beta*W);
        bRL = epsilon/nA + (1-epsilon)*bRL/sum(bRL);
        %WM policy
        W = WM(stimuli(k),:)+stick*side;
        bWM = exp(beta*W);
        bWM = epsilon/nA + (1-epsilon)*bWM/sum(bWM);
        %H policy
        W = H(stimuli(k),:)+stick*side;
        bH = exp(beta*W);
        bH = epsilon/nA + (1-epsilon)*bH/sum(bH);


        b = wWM*bWM + (1-wWM)*((1-rhoH)*bRL+rhoH*bH);


    lt = log(b(choices(k)));
    llh = llh + lt;
    ls(k)=lt;
end
end
llh = -llh;
end