function sim_data = RL2(theta,expe_data,M)
% 
% 
nA=3;
beta = 25;
for i=1:2
if M.pfixed(i)
    alphaRL(i)=M.fixedvalue(i);
else
alphaRL(i) = theta(M.thetamapping(i));%.2;
end
end

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
orRL = [r0(1),1];%[0,1];%
orWM = [r0(2),1];
    
%boum
% extracting experimental data
Allstimuli = expe_data(:,5);
Allchoices = expe_data(:,10);
Allrewards = expe_data(:,13);
AllCor = expe_data(:,12);
AllStoch = AllCor~=Allrewards;
Allsetsize = expe_data(:,3);
Allblocks = expe_data(:,2);
Allrts = expe_data(:,14);
AllaCor = expe_data(:,9);
Alliter = expe_data(:,8);
Alldelay = expe_data(:,17);
Allpcor = expe_data(:,16);

% initialize simulation data to experimental data
sim_data = expe_data;

blocks = unique(Allblocks)';

% loop over blocks
for bl = blocks
    
    Tb = find(Allblocks == bl & Allrts>.15);
    % get the block data (from participant)
    stimuli = Allstimuli(Tb);
    choices = Allchoices(Tb);
    acors = AllaCor(Tb);
    iters = Alliter(Tb);
    rewards = Allrewards(Tb);
    corrects = AllCor(Tb);
    stoch = AllStoch(Tb);
    ns = Allsetsize(Tb(1));

    % WM weight
    w = rho(ns/3);
    
    Q = (1/nA)*ones(ns,nA);
    WM = (1/nA)*ones(ns,nA);
    b= Q(1,:);
    for k = 2:length(choices)
        s = stimuli(k-1);
        if iters(k-1)==1
            choice = choices(k-1);% at first iteration, pick same action as real participant
        else
            choice = select(b);
            choices(k-1) = choice;
        end
        cor = choice == acors(k-1);
        r = cor;
        if stoch(k-1)
            r=1-r;
        end
        rewards(k-1) = r;
        corrects(k-1) = cor;

        WM = WM + forget*(1/nA - WM);
        WM(s,choice) = WM(s,choice) + alphaWM(r+1)*(orWM(r+1)-WM(s,choice));
        Q(s,choice) = Q(s,choice) + alphaRL(r+1)*(orRL(r+1)-Q(s,choice));

        side=zeros(1,nA);
        side(choice)=1;

        W = Q(stimuli(k),:)+stick*side;%+chunk*mean(Q);
        bRL = exp(beta*W);
        bRL = epsilon/nA + (1-epsilon)*bRL/sum(bRL);
        W = WM(stimuli(k),:)+stick*side;
        bWM = exp(beta*W);
        bWM = epsilon/nA + (1-epsilon)*bWM/sum(bWM);

        b = w*bWM + (1-w)*bRL;
    end
    AllCor(Tb) = corrects;
    Allrewards(Tb) = rewards;
    Allchoices(Tb) = choices;
    Alldelay(Tb) = computedelay(stimuli,rewards);
    Allpcor(Tb) = computepcor(stimuli,rewards);
    
    
    

end

sim_data(:,10) = Allchoices;
sim_data(:,12) = AllCor;
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