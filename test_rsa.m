function [acts scratchpad] = test_rsa(testpats,testtargs,scratchpad)

% Generates predictions using correlation-based rsa
%
% [ACTS SCRATCHPAD] = test_rsa(TESTPATS,TESTTARGS,SCRATCHPAD)
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

trainpats = scratchpad.model.trainpats;

nVox = size(testpats,1);

meanTestMat = repmat(mean(testpats),nVox,1); %matrix representing the mean of each test pattern
demeanedTestPats = testpats - meanTestMat;

for i=1:length(trainpats.cond)  
    
    meanTrainMat = repmat(mean(trainpats.cond{i}),nVox,1); %mean of each train pattern    
    theseDemeanedTrainPats = trainpats.cond{i} - meanTrainMat; %demean train patterns
    
    % take the median of the correlation between each test pattern and
    % members of each training class.  
    acts(i,:) = median(corr(theseDemeanedTrainPats,demeanedTestPats)); 
  
end