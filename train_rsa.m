

function [scratchpad] = train_rsa(trainpats,traintargs,in_args,cv_args)

% split training patterns by condition, in preparation for formatting for
% test_rsa

for i=1:size(traintargs,1)
   model.trainpats.cond{i} = trainpats(:,traintargs(i,:)==1);  
end

scratchpad.model = model;
