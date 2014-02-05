function p = fisherConj( pVec )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

C = -2*sum(log(pVec));
df = 2*length(pVec);

p = 1-chi2cdf(C, df);

end

