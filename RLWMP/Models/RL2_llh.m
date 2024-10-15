function llh = RL2_llh(theta,data,M)
%
%
% if nargin<4
% plotting=0;
% end

nA = 3;
beta = 25;
%RL params
for i=1:2
    if M.pfixed(i)
        alphaRL(i)=M.fixedvalue(i);
    else
        alphaRL(i) = theta(M.thetamapping(i));%.2;
    end
end
% WM params
alphaWM = theta(M.thetamapping(3:4));%.2;
for i=1:2
    if M.pfixed(4+i)
        rho(i) =M.fixedvalue(4+i);
    else
        rho(i) = theta(M.thetamapping(4+i));
    end
end
rho(2) = rho(1)*rho(2);
if M.pfixed(7)
    forget =M.fixedvalue(7);
else
    forget = theta(M.thetamapping(7));
end
if M.pfixed(8)
    epsilon = M.fixedvalue(8);
else
    epsilon = theta(M.thetamapping(8));%
end
if M.pfixed(9)
    stick = M.fixedvalue(9);
else
    stick = theta(M.thetamapping(9));
end
if M.pfixed(10)
    r0(1) = M.fixedvalue(10);
else
    r0(1) = theta(M.thetamapping(10));
end
if M.pfixed(11)
    r0(2) = M.fixedvalue(11);
else
    r0(2) = theta(M.thetamapping(11));
end
orRL = [r0(1),1];
orWM = [r0(2),1];

%alphaRL = alpha;%[alpham(1) alpha(1)];
%alphaWM = alpham;%[alpham(2) alpha(2)];

Allstimuli = data(:,1);
Allchoices = data(:,2);
Allrewards = data(:,3);
Allsetsize = data(:,4);
Allblocks = data(:,5);
Allrts = data(:,6);

blocks = unique(Allblocks)';

llh = 0;

for bl = blocks
    Tb = find(Allblocks == bl & Allrts>.15);
    stimuli = Allstimuli(Tb);
    choices = Allchoices(Tb);
    rewards = Allrewards(Tb);
    ns = Allsetsize(Tb(1));
    
    % WM weight
    w = rho(ns/3);
    
    Q = (1/nA)*ones(ns,nA);%[.5 .5;.5 .5];
    WM = (1/nA)*ones(ns,nA);%[.5 .5;.5 .5];
    lt = log(1/nA);
    llh = llh + lt;
    
    for k = 2:length(choices)
        s = stimuli(k-1);
        choice = choices(k-1);
        r = rewards(k-1);
        
        WM = WM + forget*(1/nA - WM);
        WM(s,choice) = WM(s,choice) + alphaWM(r+1)*(orWM(r+1)-WM(s,choice));
        Q(s,choice) = Q(s,choice) + alphaRL(r+1)*(orRL(r+1)-Q(s,choice));
        
        side=zeros(1,nA);
        side(choice)=1;
        
        % RL policy
        W = Q(stimuli(k),:)+stick*side;% sticky choice
        bRL = exp(beta*W);% softamx
        bRL = epsilon/nA + (1-epsilon)*bRL/sum(bRL);% blend with epsilon greedy
        % WM policy
        W = WM(stimuli(k),:)+stick*side;
        bWM = exp(beta*W);
        bWM = epsilon/nA + (1-epsilon)*bWM/sum(bWM);
        
        b = w*bWM + (1-w)*bRL;
        lt = log(b(choices(k)));
        if isnan(lt)
            Q
            bRL
            W
            bWM
            boum
        end
        llh = llh + lt;
        ls(k)=lt;
    end
end
llh = -llh;
end