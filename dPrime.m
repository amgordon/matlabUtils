
function [dPrime c] = dPrime(hitrate, farate)
% [dPrime c] = calc_dPrime;
% JC 03/01/06


dPrime = norminv(hitrate,0,1) - norminv(farate,0,1);
c = -0.5 * ( norminv(hitrate,0,1) + norminv(farate,0,1));
