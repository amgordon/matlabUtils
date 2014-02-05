function [acts scratchpad] = test_liblinear(testpats,testtargs,scratchpad)

% Generates predictions using a trained logistic regression model
%
% [ACTS SCRATCHPAD] = TEST_RIDGE(TESTPATS,TESTTARGS,SCRATCHPAD)
%
% License:
%=====================================================================
%
% This is part of the Princeton MVPA toolbox, released under
% the GPL. See http://www.csbmb.princeton.edu/mvpa for more
% information.
% 
% The Princeton MVPA toolbox is available free and
% unsupported to those who might find it useful. We do not
% take any responsibility whatsoever for any problems that
% you have related to the use of the MVPA toolbox.
%
% ======================================================================

if scratchpad.constant
  testpats = [testpats; ones(1,cols(testpats))];
end

% output predictions goes into "ACTS"
acts = zeros(size(testtargs));

%[nConds nTimepoints] = size(testtargs);
nConds = rows(testtargs);


thisTest = sparse(double(testpats));
% 
% if size(testtargs,1)==2
%     theseTestLabels = 2*(testtargs(1,:)') - 1; %must use -1 and 1 labels
% else
%     for i=1:size(testtargs,2)
%         theseTestLabels(i,1) = find(testtargs(:,i))';
%     end
% end

% re-order to account for liblinear's annoying tendency to order classes by
% the order in which they appear in the training labels (as opposed toa more sensible option like 1,
% 2, 3)

testTargsReoriented = testtargs;%(scratchpad.classOrientation,:);

for i=1:size(testTargsReoriented,2)
    theseTestLabels(i,1) = find(testTargsReoriented(:,i))';
end

if strcmp(scratchpad.classType, 'svm')
    [theseLabels acc probVals]=svmpredict(theseTestLabels, thisTest', scratchpad.model);
    acts = ([probVals -1*probVals]);

else
    [theseLabels acc probVals]=predict(theseTestLabels, thisTest', scratchpad.model, '-b 1');
    acts = nan(size(probVals));
    for i = 1:length(scratchpad.classOrientation)
        acts(:,i) = probVals(:,scratchpad.classOrientation(i),:);
    end
end



acts = acts';
%probVals = exp(logits) ./ (1 + exp(logits));

%  if size(probVals,2)==2
%      if scratchpad.liblinOrientation==1
%          acts = [probVals'; 1-probVals'];
%      else
%          acts = [1-probVals'; probVals'];
%      end
%  else
%     
%     acts = probVals'./repmat(sum(probVals'),size(probVals,2),1);
%  end

% scratchpad.w = [scratchpad.logreg.betas, -1*scratchpad.logreg.betas];



%             % test it
%             thisTest = X_svm(testIdx,:);
%             theseTestLabels = Y_svm(testIdx,:);
%             [theseLabels,~]=predict(theseTestLabels, thisTest, model, S.predictOpts);
%             
%             % put the results in the acts file.;
%             guesses(testIdx) = theseLabels;