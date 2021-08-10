function [stegoB] = f_emb_filter(cover,pixel_change, rho,seed, w, payload)
params.w=w;%����仯���챶��

params.H=0;

wetCost = 10^10;


%% Get embedding costs

% inicialization

cover = double(cover);


rhoP1 = rho;

rhoM1 = rho;

rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value

rhoM1(cover==0) = wetCost; % do not embed -1 if the pixel has min value

rhoP1(pixel_change==1)=(rhoP1(pixel_change==1))./params.w;

rhoM1(pixel_change==-1)=((rhoM1(pixel_change==-1)))./params.w;

stegoB = f_EmbeddingSimulator_seed(cover, rhoP1, rhoM1, floor(payload*numel(cover)), seed);


