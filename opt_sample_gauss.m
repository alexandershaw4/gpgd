function [X,F] = opt_sample_gauss(fun,x0,V,y,N,NN,doplot)

if nargin < 7; doplot = 0; end
if nargin < 6; NN = 10; end
if nargin < 5; NN = 32; end

f  = @(x) sum( (spm_vec(y) - spm_vec(fun(x))).^2);
e0 = f(x0);
ex = e0;
ee = [];

if isvector(V); V = full(diag(V)); end

for j = 1:NN

    %fprintf('Main sampling loop: %d/%d\n',j,NN);
    list = [];ne = [];
    
    for i = 1:N
        R = mvnrnd(x0,V)';
    
        if f(R) < ex
            list = [list R];
            ne   = [ne; f(R)];
        end
    
    end
    
    if any(ne)
        % assess best performing;
        [~,I] = min(ne);
    
        % compute new parameters and threshold 
        x0 = list(:,I);
        ex = ne(I);
        ee = [ee(:); ex];

        fprintf('Some improvment on this loop (dF = %d)\n',ex-e0);

        if doplot
            w = 1:length(y);
            subplot(121);plot(w,y,':',w,fun(x0)); 
            subplot(122);plot(ee); drawnow;
        end
    else

    end

end

F = f(x0);
X = x0;

end
