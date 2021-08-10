function [C_STRUCT, C_SPATIAL, C_COEFFS, C_QUANT] = read_jpeg(coverPath)
addpath(fullfile('.','JPEG_Toolbox'));
C_SPATIAL = double(imread(coverPath));
C_STRUCT = jpeg_read(coverPath);
C_COEFFS = C_STRUCT.coef_arrays{1};
C_QUANT = C_STRUCT.quant_tables{1};
end
