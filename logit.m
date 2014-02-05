function [ out ] = logit(prob)
%Take the logit of a p-value           

out = log(prob ./ (1 - prob));

end

