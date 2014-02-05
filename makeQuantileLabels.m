function labs = makeQuantileLabels(vec,n)

if n==2
    q_h = quantile(vec,3);
    q = q_h(2);
else
    q = quantile(vec,n-1);
end

[labHist, labs]= histc(vec, [min(vec) q Inf ]);

labHist = labHist(1:(end-1));
if (max(labHist) - min(labHist))>1
   warning('labels not uniformly distributed, possibly because of duplicate values'); 
end

