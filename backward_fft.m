function f=backward_fft(cf)
% iFFT caller to inverse FFT a field.
    tn_phi=size(cf,2);
    n_phi=(tn_phi-1)*2;
    cf_full=[cf zeros(size(cf,1),n_phi-tn_phi)];
    f=ifft(cf_full,[],2,'symmetric')*n_phi;
end