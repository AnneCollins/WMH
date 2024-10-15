% Anne Collins, UC Berkeley
% annecollins@berkeley.edu, 2024
% Code for paper "RL or not RL? Parsing the processes that support human
%reward-based learning."

% This script performs runs and plots simulations for Fig. 4.
% with 1000 simulations, this script runs in ~245s on my computer (MacBook Apple M1 Pro).

clear all
tic
%% set up
ntrials = 50;
niter = 1000;% set to a smaller number for debugging. 
%% model params
betas = [2 5 8];% different softmax beta values
alphaH = [.1 .1];%H agent learning rate (alpha-, alpha+)
decay = .01;% forgetting
rhos = [0 .1 .5 .9 1];% WM weight
%% experiment setup
noises = [0:.05:1];% noise in p(reward=1|correct)=p(reward=0|incorrect).

% colors for figures
%col = [zeros(5,1) (1:-.25:0)' zeros(5,1)];

figure

%% run for RL and for H agents
for RL_or_Hebb=0:1
    col = zeros(5,3);
    % set up different figures for the two agents.
    col(:,2+RL_or_Hebb,:) = (1:-.25:0)';
    kb=0;
    % loop over different values of softmax beta
    for beta = betas
        kb = kb+1;
        rh=0;
        % different subplot per beta
        subplot(2,length(betas)+1,kb+RL_or_Hebb*(length(betas)+1))
        % loop over WM weight in policy
        for rho = rhos%(1);
            rh = rh+1;
            %%            
            kn =0;
            % loop over reward stochasticity level
            for noise = noises;
                kn = kn+1;
                % loop over simulation iterations
                for iter = 1:niter
                    % initialize WM
                    WM = [0 0];
                    %initialize Hebb/RL weights
                    W_weights = [0,0]+.5;
                    
                    Data = [0 nan nan nan WM W_weights];
                    % run through trials
                    for t=1:ntrials
                        % WM policy
                        p_WM = 1/(1+exp(beta*(WM(1)-WM(2))));
                        % RL/H policy
                        p_H = 1/(1+exp(beta*(W_weights(1)-W_weights(2))));
                        % mixture policy
                        p = rho*p_WM + (1-rho)*p_H;
                        % store for plotting
                        policy(t,:) = [p_WM p_H p];
                        % make a choice
                        choice = 1+(rand<p);
                        % correct choice is choice 2
                        correct = choice==2;
                        % stochastic reward
                        if rand<noise
                            reward=1-correct;
                        else
                            reward = correct;
                        end
                        
                        %WM updated [remembers each choice's last reward]
                        WM(choice)=reward;
                        %Hebb/RL weights
                        W_weights = (1-decay)*W_weights;
                        if RL_or_Hebb
                            % Hebb: RPE is 1-W
                            W_weights(choice) = W_weights(choice) + alphaH(1+reward)*(1-W_weights(choice));
                        else
                            % Hebb: RPE is r-W
                            W_weights(choice) = W_weights(choice) + alphaH(1+reward)*(reward-W_weights(choice));
                        end
                        % store relevant variables
                        Data = [Data;[t choice correct reward WM W_weights]];
                    end
                    % store learning curve
                    perf(iter,:) = Data(2:end,3);
                    % store policies
                    policies(iter,:,:)= policy;
                    
                end
                
                %%
                % figure;
                % subplot(2,1,1)
                % errorbar(mean(perf),std(perf)/sqrt(iter))
                % subplot(2,1,2)
                % hold on
                % for i=1:3
                %     errorbar(mean(policies(:,:,i)), std(policies(:,:,i))/sqrt(iter))
                % end
                % legend('WM','Hebb','all')
                
                % store final learned H/RL policy
                p_end(:,kn) = policies(:,end,2);
                
            end
            %%
            % plot final learned policy
            hold on
            errorbar(1-noises,mean(p_end),std(p_end)/sqrt(size(p_end,1)),'color',col(rh,:),'linewidth',1.4,'CapSize',1.4)
            xlabel('p(r=1|A)')
            
            if RL_or_Hebb
                ylabel('\pi_H(A)')
            else
                ylabel('\pi_{RL}(A)')
            end
        end
        set(gca,'fontsize',14)
        title(['\beta=',num2str(beta)])
        ylim([0 1])
    end
legend('\rho_{WM}=0','\rho_{WM}=.1','\rho_{WM}=.5','\rho_{WM}=.9','\rho_{WM}=1')
end
toc