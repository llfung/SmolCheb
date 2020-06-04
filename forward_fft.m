function cf=forward_fft(f)
% FFT caller to FFT a field.
    n=size(f,2);
    tn_phi=(n/2+1);
    cf_full=fft(f,[],2)/n;
    cf=cf_full(:,1:tn_phi);
end