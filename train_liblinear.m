
function [scratchpad] = train_liblinear(trainpats,traintargs,in_args,cv_args)
%
%
% Function written by Alan Gordon to fit mvpa toolbox conventions

v = (double(trainpats'));

% if size(traintargs,1)==2
%     choice = 2*(traintargs(1,:)') - 1; %must use -1 and 1 labels
% else
for i=1:size(traintargs,2)
    choice(i,1) = find(traintargs(:,i))';
end
% end

voxel_num = size(v,2);
lambda_beta = 0;


doRadialSearch = ~(strcmp(in_args.classType,'libLin') || isempty(in_args.radialBasisSelection));
%%  pick the optimal cost parameter

choice_set = unique(choice);
%MUST MAKE THIS GENERAL FOR >2 CLASSES
if in_args.chooseOptimalPenalty
    
    
    idx.c1 = find(choice==choice_set(1));
    idx.c2 = find(choice==choice_set(2));
    
    guesses = nan(size(choice));
    
    %loop through indices, one at a time
    randomNFold1 = ceil(shuffle(1:length(idx.c1))/(length(idx.c1)/in_args.nFoldsPenaltySelection));
    randomNFold2 = ceil(shuffle(1:length(idx.c2))/(length(idx.c2)/in_args.nFoldsPenaltySelection));
    for s = 1:in_args.nFoldsPenaltySelection %assumes balanced data
        
        thisIdx1 = randomNFold1==s;
        
        omitc1 = idx.c1(randomNFold1==s);
        omitc2 = idx.c2(randomNFold2==s);
        
        theseOmitted = [omitc1; omitc2];
        
        thisChoice = choice;
        thisChoice(theseOmitted) = [];
        
        thisV = v;
        thisV(theseOmitted,:) = [];
        
        
        for i = 1:length(in_args.penaltyRange)
            l = in_args.penaltyRange(i);
            
            if doRadialSearch
                rSet = in_args.radialBasisSelection;
            else
                rSet = 0;
            end
            
            for r = 1:length(rSet)
                thisR = rSet(r);
                if strcmp(in_args.classType,'libLin')
                    trainOpts_orig = in_args.libLin ;
                    trainOptsOptimize = [trainOpts_orig ' -c ' num2str(l)];
                    m = train(thisChoice, sparse(thisV), trainOptsOptimize);
                    [theseLabels,~]=predict(choice(theseOmitted), sparse(v(theseOmitted,:)), m);
                elseif strcmp(in_args.classType,'svm')
                    trainOpts_orig = in_args.libsvm ;
                    trainOptsOptimize = [trainOpts_orig ' -c ' num2str(l) ' -r ' num2str(thisR)];
                    m = svm_train(thisChoice, thisV, trainOptsOptimize);
                    testChoice = choice(theseOmitted);
                    [theseLabels,~]=svmpredict(testChoice, v(theseOmitted,:), m);
                    
                end
                
                guesses(theseOmitted,i,r) = theseLabels;
                
            end
        end
    end
    choiceMat = repmat(choice,[1,length(in_args.penaltyRange), length(rSet)]);
    % performance measures
    perf.tp = sum(guesses == choice_set(1) & choiceMat == choice_set(1));   % true pos
    perf.fp = sum(guesses == choice_set(1) & choiceMat == choice_set(2));  % false pos
    perf.fn = sum(guesses == choice_set(2) & choiceMat == choice_set(1));   % false neg
    perf.tn = sum(guesses == choice_set(2) & choiceMat == choice_set(2));  % true neg
    
    perf.Precision = perf.tp./(perf.tp+perf.fp);
    perf.Recall = perf.tp./(perf.tp+perf.fn);
    
    perf.TrueNegRate = perf.tn./(perf.tn+perf.fp);
    %perf.Accuracy = (perf.tp+perf.tn)./ ...
    %    (perf.tp+perf.tn+perf.fp+perf.fn);
    
    perf.Accuracy = ((perf.tp)./(perf.tp + perf.fn) + (perf.tn)./(perf.tn + perf.fp))*.5;
    
    perf.F_Score = 2.*perf.Precision.*perf.Recall./ ...
        (perf.Precision+perf.Recall);
    
    % use F score as the performance measure to select the optimal
    % parameter
    perfParam = perf.Accuracy;
    
    optPerf = max(max(perfParam));
    perfParam = squeeze(perfParam);
    
    if doRadialSearch
        [idx.optPerf1 idx.optPerf2] = find(perfParam==optPerf);
        opt_penalty = in_args.penaltyRange(idx.optPerf1(1));
        opt_r = in_args.radialBasisSelection(idx.optPerf2(1));
    else
        idx.optPerf = find(perfParam==optPerf);
        opt_penalty = in_args.penaltyRange(idx.optPerf(1));
    end
else
    opt_penalty = in_args.penalty;
end

scratchpad.opt_penalty = opt_penalty;
%% classify with the optimal penalty param, established in a non-biased fashion by cross-validating the training data


% train it, with leave one out cross validation
if strcmp(in_args.classType,'svm')
    trainOpts_orig = in_args.libsvm;
    trainOpts = [trainOpts_orig ' -c ' num2str(opt_penalty) ' -r ' sprintf('%f', opt_r)];
    model = svm_train(choice, sparse(v), trainOpts);
    scratchpad.classOrientation = model.Label';
else
    trainOpts_orig = in_args.libLin ;
    trainOpts = [trainOpts_orig ' -c ' num2str(opt_penalty)];
    model = train(choice, sparse(v), trainOpts);
    scratchpad.classOrientation = model.Label';
    scratchpad.logreg.betas = [model.w(end) model.w(1:(end-1))];
    scratchpad.w(:,scratchpad.classOrientation(1),:) = scratchpad.logreg.betas';
    scratchpad.w(:,scratchpad.classOrientation(2),:) = -1*scratchpad.logreg.betas';
end
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
scratchpad.classType = in_args.classType;
scratchpad.constant = in_args.constant;
scratchpad.choice = choice;
scratchpad.model = model;


