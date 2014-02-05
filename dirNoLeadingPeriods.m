function [ d ] = dirNoLeadingPeriods( str )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

d = dir(str);
d = d(find(cellfun(@(x) ~strcmp(x(1),'.'), {d(:).name})));

end

