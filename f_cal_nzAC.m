function [nzAC] = f_cal_nzAC(C_COEFFS)
nzAC = nnz(C_COEFFS)-nnz(C_COEFFS(1:8:end,1:8:end));
end
