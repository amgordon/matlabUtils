function ste = steRMANOVA (DV, IV, subs)

IV_set = unique(IV);
IVLab = nan(size(IV));
for i=1:length(IV_set)
    IVLab(IV==IV_set(i)) = i;
end
anovaStats = ar_rmanova1([DV IVLab subs]);
ste = sqrt(anovaStats.MSE/length(subs));