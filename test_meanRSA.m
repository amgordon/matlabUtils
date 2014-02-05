function [acts scratchpad] = test_meanRSA(testpats,testtargs,scratchpad)


%acts = corr(testpats,[scratchpad.meanPatClass1 scratchpad.meanPatClass2])';


acts(1,:) = mean(corr(testpats, scratchpad.patClass1),2)';
acts(2,:) = mean(corr(testpats, scratchpad.patClass2),2)';