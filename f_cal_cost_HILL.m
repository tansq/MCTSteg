function [cost,r] = f_cal_cost_HILL(cover)
%Get filter
HF=[-1 2 -1;2 -4 2;-1 2 -1];
H2 =  fspecial('average',[3 3]);
%% Get cost
cover=double(cover);
sizeCover=size(cover);
padsize=max(size(HF));
coverPadded = padarray(cover, [padsize padsize], 'symmetric');% add padding
R = conv2(coverPadded,HF, 'same');%mirror-padded convolution
W=conv2(abs(R),H2,'same');
% correct the W shift if filter size is even
 if mod(size(HF, 1), 2) == 0, W = circshift(W, [1, 0]); end;
 if mod(size(HF, 2), 2) == 0, W = circshift(W, [0, 1]); end;
  % remove padding
 W = W(((size(W, 1)-sizeCover(1))/2)+1:end-((size(W, 1)-sizeCover(1))/2), ((size(W, 2)-sizeCover(2))/2)+1:end-((size(W, 2)-sizeCover(2))/2));
 r=W;
 cost=1./(W+10^(-10)); 
 wetCost = 10^10;
% compute embedding costs \rho
rhoA = cost;
rhoA(rhoA > wetCost) = wetCost; % threshold on the costs
rhoA(isnan(rhoA)) = wetCost; % if all xi{} are zero threshold the cost

HW =  fspecial('average', [15, 15]) ;
cost = imfilter(rhoA, HW ,'symmetric','same');
 