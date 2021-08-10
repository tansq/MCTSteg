function [path] = write_jpeg(S_COEFFS, path)
addpath(fullfile('.','JPEG_Toolbox'));
C_STRUCT = jpeg_read(path);
C_STRUCT.coef_arrays{1} = S_COEFFS;
jpeg_write(C_STRUCT, path);
end

