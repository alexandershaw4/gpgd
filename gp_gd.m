function [X,F] = gp_gd(fun,x0,V,y)
% A sort of Bayesian Gauss-Newton / gradient descent routine;
%
%      [X,F] = gp_gd(fun,x0,V,y)
%
% fun = function, taking parameters x0; f(x0)
% V   = vector matching length x0 with initial deltas for finite difference 
% routine; set element to 0 to fix a parameter.
% y   = data to fit by minimising SSE: sum( (y(:) - f(x0)).^2 )
%
% Acts like a normal Gauss-Newton, except the usual (regularised) step; 
%       step     = (I+J'*J)\(J'*r)
%       p(n + 1) = p(n) + step 
%
% is instead replaced with a Bayesian step returning the mean and
% variance;
%       l = (p1.^2)./(p2.^2);
%       M = (l*I+Jg'*Jg)\(Jg'*r);
%       C = inv(p1.^(-2)*(Jg'*Jg)+p2.^(-2).*I)
% 
% where p1 and p2 are explicitly optimised used fminsearch. This is
% repeated for each element of the Jacobian, converting each vector J(i,:)
% to a Gaussian process matrix.
%
% The result is a mean and variance for each parameter on each iteration.
% Instead of assuming the mean and using p(n + 1) = p(n) + mean, we instead
% sample from the identified Gaussian winow for each parameter and accept the
% best function value.

% Objective
f = @(p) sum((spm_vec(y) - spm_vec(fun(p))).^2);
s = 1/32;
e = f(x0);
N = 10;

for i = 1:N

    % Gradients & Residual
    J = jaco_mimo_par(fun,x0,V,0,1); 
    J = cat(2,J{:})';
    r = spm_vec(y) - spm_vec(fun(x0));
    
    % Normalise by vector norms
    for j = 1:size(J,1)
        J(j,:) = J(j,:) ./ norm(J(j,:));
    end
    
    J = denan(J);
    r = r./norm(r);
    
    % (Gaussian) Step estimate - similar to GN but with variance estimated
    % by a Gaussian Process model
    [bx,bvx,mb,vb] = agaussreg(J',r);
    
    % Ensure only using params with prior variance > 0
    ip    = find(V);
    b     = bx*0;
    b(ip) = bx(ip);
    
    % Sample from predicted Gaussian window
    sfun  = @(v) fun(x0 - (s)*v);
    [X,F] = opt_sample_gauss(sfun,b,vb*vb',y,10);
    
    % Compute actual step
    dx = x0 - (s)*X;
    
    % Compare
    if f(dx) < f(x0)
        x0 = dx;
        e  = [e; f(x0)];
    
        w  = 1:length(y);
        subplot(121),plot(w,y,':',w,fun(x0));
        subplot(122),plot(e); drawnow;
    
        fprintf('Improvement on iteration %d\n',i);
    
    else
        s = s / 2;
    
        fprintf('No improvement on iteration %d\n',i);
    end
end

X = x0;
F = e(end);

