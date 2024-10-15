function [LC] = analyzeBehavior(X)

T = find(X(:,14)>0.15);

block = X(T,2);
ns = X(T,3);
iter = X(T,8);
cor = X(T,12);
rew = X(T,13);
pcor = X(T,16);
pinc = iter - pcor - 1;
delay = X(T,17);


setsizes = unique(ns');
iters = unique(iter');

for n = setsizes(setsizes>1)
    for i = 1:10
        T = find(iter==i & ns==n);
        LC(n,i) = mean(cor(T));
    end
    
end


    

end

