
function [scratchpad] = train_liblinear(trainpats,traintargs,in_args,cv_args)
% 
%   
% Function written by Alan Gordon to fit mvpa toolbox conventions

traintargs = 2-traintargs(1,:);
 
 trainPats1 = trainpats(:,traintargs==1);
 trainPats2 = trainpats(:,traintargs==2);
% 
% scratchpad.meanPatClass1 = mean(trainPats1,2);
% scratchpad.meanPatClass2 = mean(trainPats2,2);

scratchpad.patClass1 = trainPats1;
scratchpad.patClass2 = trainPats2;



