function testconsistency(curr_model)

 
 nfixed = sum(curr_model.pfixed);
 nfree = max(curr_model.thetamapping);
 nnotfixed = sum(~isnan(curr_model.thetamapping));
 nMin = length(curr_model.pMin);
 nMax =length(curr_model.pMax);
 
if nfree~=nMax% for K
     nfree
     nMax
     error('wrong number of free parameters vs. pMax')
end

if nfree~=nMin% for K
     nfree
     nMin
     error('wrong number of free parameters vs. pMin')
end

if nfixed+nnotfixed~=length(curr_model.pnames)% for K
     nnotfixed
     nfixed
     length(curr_model.pnames)
     error('wrong number of free parameters vs. fixed parameters')
end