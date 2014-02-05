function [ C ] = mat2CellVals( vec )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
for i=1:length(vec)
    C{i} = vec(i);
end

