
function [scratchpad] = train_liblinear(trainpats,traintargs,in_args,cv_args)
% 
%   
% Function written by Alan Gordon to fit mvpa toolbox conventions
 
v = sparse(double(trainpats'));

if (~isempty(strfind(in_args.libsvm, '-s 3')) || ~isempty(strfind(in_args.libsvm, '-s 4')))
    choice = traintargs';
else
    if size(traintargs,1)==2
        choice = 2*(traintargs(1,:)') - 1; %must use -1 and 1 labels
    else
        for i=1:size(traintargs,2)
            choice(i,1) = find(traintargs(:,i))';
        end
    end
end

voxel_num = size(v,2);
lambda_beta = 0;

trainOpts_orig = in_args.libsvm ;
%%  pick the optimal cost parameter

%MUST MAKE THIS GENERAL FOR >2 CLASSES
if in_args.chooseOptimalPenalty
    for i = 1:length(in_args.penaltyRange)
        idx.c1 = find(choice==1);
        idx.c2 = find(choice==-1);
        
        guesses = nan(size(choice));
        
        %loop through indices, one at a time
        for s = 1:length(idx.c1) %assumes balanced data
            omitc1 = idx.c1(s);
            omitc2 = idx.c2(s);
            
            theseOmitted = [omitc1 omitc2];
            
            thisChoice = choice;
            thisChoice(theseOmitted) = [];
            
            thisV = v;
            thisV(theseOmitted,:) = [];
            
            l = in_args.penaltyRange(i);
            trainOptsOptimize = [trainOpts_orig ' -c ' num2str(l)];
            m = svm_train(thisChoice, thisV, trainOptsOptimize);
           
            
            [theseLabels,~]=svmpredict(choice(theseOmitted), v(theseOmitted,:), m);
            
            guesses(theseOmitted) = theseLabels;
            
        end
        
        % performance measures
        perf.tp = sum(guesses == 1 & choice == 1);   % true pos
        perf.fp = sum(guesses == 1 & choice == -1);  % false pos
        perf.fn = sum(guesses ==-1 & choice == 1);   % false neg
        perf.tn = sum(guesses ==-1 & choice == -1);  % true neg
        
        perf.Precision = perf.tp/(perf.tp+perf.fp);
        perf.Recall = perf.tp/(perf.tp+perf.fn);
        
        perf.TrueNegRate = perf.tn/(perf.tn+perf.fp);
        perf.Accuracy = (perf.tp+perf.tn)/...
            (perf.tp+perf.tn+perf.fp+perf.fn);
        
        perf.F_Score = 2*perf.Precision*perf.Recall/...
            (perf.Precision+perf.Recall);
        
        % use F score as the performance measure to select the optimal
        % parameter
        perfParam(i) = perf.F_Score;
        
    end
    
    [optPerf, idx.optPerf] = max(perfParam);
    opt_penalty = in_args.penaltyRange(idx.optPerf);
else
    opt_penalty = in_args.penalty;
end

%% classify with the optimal penalty param, established in a non-biased fashion by cross-validating the training data
trainOpts_orig = in_args.libsvm ;
trainOpts = [trainOpts_orig ' -c ' num2str(opt_penalty)];

% train it, with leave one out cross validation
model = svm_train(choice, v, trainOpts);

% liblinear naturally picks the first choice direction as the first
% choice.  This differs from mvpa toolbox, which keeps the '1' labels as
% the first choice regardless of what the first presented trial label is.
% This variable ensures consistency across these toolboxes.

choice_set = unique(choice);
for i = 1:length(choice_set)
    thisChoice = choice_set(i);
    findChoice = find(choice==thisChoice);
    earliestInstanceOfChoice(i) = findChoice(1);
end

scratchpad.classOrientation = model.Label';

%scratchpad.logreg.betas = [model.w(end) model.w(1:(end-1))];

% if size(unique(choice),1)==2
%     scratchpad.liblinOrientation = choice(1);
%     
%     if scratchpad.liblinOrientation==1
%         scratchpad.logreg.betas(:,1) = [model.w(end) model.w(1:(end-1))];
%     else
%         scratchpad.logreg.betas(:,1) = [-model.w(end) -model.w(1:(end-1))];
%     end
% else
%     scratchpad.logreg.betas = model.w;
% end

scratchpad.constant = in_args.constant;
scratchpad.choice = choice;
scratchpad.model = model;
scratchpad.args = in_args;
