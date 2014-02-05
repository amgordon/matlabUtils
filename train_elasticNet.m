
function [scratchpad] = train_elasticNet(trainpats,traintargs,in_args,cv_args)
% 
%   
% Function written by Alan Gordon to fit mvpa toolbox conventions
 
v = sparse(double(trainpats'));
choice = 2*(traintargs(1,:)') - 1; %must use -1 and 1 labels


%%  pick the optimal cost parameter


glmnetoptions = glmnetSet();
glmnetoptions.alpha = S.glmnet.alpha;
glmnetoptions.nlambda = S.glmnet.nlambda;

m = cvglmnet(v,choice,S.nXvals,foldIdAllTrain,'class','binomial',glmnetoptions,0);

glmnetoptions = glmnetSet();
glmnetoptions.lambda =  m.lambda_1se;
glmnetoptions.alpha = S.glmnet.alpha;
%glmnetoptions.alpha = m.alpha_min;

model = glmnet(XallTrain, YallTrain, 'binomial', glmnetoptions);

scratchpad.constant = in_args.constant;

scratchpad.model = model;

