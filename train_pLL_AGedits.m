
function [scratchpad] = train_pLL_AGedits(trainpats,traintargs,in_args,cv_args)
% 
%   
% Function written by Roozbeh Kiani, modified by Alan Gordon to fit mvpa
% toolbox conventions
 
v = trainpats';
choice = traintargs(1,:)';
p = in_args.penalty;

voxel_num = size(v,2);
lambda_w = repmat(p,voxel_num,1);
lambda_beta = 0;


if in_args.prefitWeights
    %fit choices with ridge logistic to develop a guess for the initial values of w
    %and beta
    S = train_logreg(v', choice', in_args);
    g_ = S.logreg.betas;
    guess_w = g_(2:end);
    guess_beta = g_(1);
else
    guess_w = zeros(size(lambda_w));
    guess_beta = 0;
end

in_args.penalty = median(lambda_w);



    
    model = struct('init', struct('w',guess_w,'beta',guess_beta), ...
                   'final', struct('w',[],'beta',[]), ...
                   'fit', struct('exitflag',[],'output',[]), ...
                   'LL', NaN, ...
                   'evidence', [], ...
                   'choice', [], ...
                   'rt', []);

               guess_param = [guess_w'  guess_beta]';
    
    iter_count = 0;
    options = optimset('Display', 'off', ...
                       'FunValCheck', 'on', ...
                       'MaxFunEval', 300*(length(guess_w)+length(guess_beta)), ...
                       'MaxIter', 300*(length(guess_w)+length(guess_beta)), ...
                       'TolFun', 1e-6, ...
                       'TolX', 1e-6, ...
                       'GradObj', 'on', ...
                       'Hessian', 'on');
    [param, fval, exitflag, output] = fminunc(@errFunc, guess_param, options);
    model.final.w = param(1:end-1);
    model.final.beta = param(end);
    model.fit.exitflag = exitflag;
    model.fit.output = output;
    model.LL = -fval;
    model.evidence = v*model.final.w + model.final.beta(1);
    model.choice = 1./(1+exp(-model.evidence));
    
    if nargin>=5 && ~isempty(pred_v)
        e = pred_v*model.final.w + model.final.beta(1);
        pred_c = 1./(1+exp(-e));
    else
        pred_c = [];
    end
    
    scratchpad.logreg.betas(:,1) = [param(end); param(1:end-1)];
    scratchpad.logreg.betas(:,2) = [-param(end); -param(1:end-1)];
    
    scratchpad.constant = in_args.constant;
    
    %% errFunc
    function [f, g, h, pred_c] = errFunc(param)
        iter_count = iter_count + 1;
        w = param(1:end-1);
        beta = param(end);
        
        e = v*w + beta(1);
        
        pred_c = 1./(1+exp(-e));
        
        ll1 = choice'*log(pred_c) + (1-choice)'*log(1-pred_c);
        if ~isfinite(ll1)
            ll1 = choice'*log(max(1e-8,pred_c)) + (1-choice)'*log(max(1e-8,1-pred_c));
        end
        dw1 = ((choice-pred_c)'*v)';
        db1 = sum(choice-pred_c);
        A = -pred_c.*(1-pred_c);
        dwdw1 = v'*diag(A)*v;
        dbdb1 = zeros(length(beta));
        dbdb1(1) = sum(A);
        dwdb1 = [v'*A zeros(length(w),length(beta)-1)];
        dbdw1 = dwdb1';
        
        ll2 = -(lambda_w.*w)'*w -(lambda_beta.*beta)'*beta;
        dw2 = -2*lambda_w.*w;
        db2 = -2*lambda_beta.*beta;
        h2 = -2*diag([lambda_w; lambda_beta]);
        
        f = -(ll1+ll2);
        g = -[dw1+dw2
            db1+db2];
        h = -([dwdw1   dwdb1
            dbdw1   dbdb1]+h2);
    end
end

