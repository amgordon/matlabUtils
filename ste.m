function ste = ste(vec)

ste = sqrt(nanvar(vec))./sqrt(sum(~isnan(vec)));

