clear,close, clc
%E2E-Fusion: A Self-supervised Deep Denoiser for Hyperspectral and Multispectral Image Fusion 
% ---------------------------------------------------------------------
% For E2E-Fusion method, see more details in the paper:
%
% Zhicheng Wang, Michael K. Ng, Joseph Michalski, and Lina Zhuang,
% "A Self-Supervised Deep Denoiser for Hyperspectral and Multispectral Image Fusion,"
% in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2023.3303921.
%  
%% -------------------------------------------------------------------------
%
% Copyright (Sep. 2022):
% Authors: Zhicheng Wang (wangzc22@connect.hku.hk)
%         &
%         Lina Zhuang (linazhuang@qq.com)
%
%
% E2E-Fusion is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ---------------------------------------------------------------------
%%
addpath ./data
addpath ./function
load F_wadc.mat
load WaDC_part1.mat
load WaDC_part2.mat
img = double(cat(3,imgpart1,imgpart2));
S= img;
F=sp_matrix';
[M,N,L] = size(S);

%%  simulate LR-HSI
S_bar = hyperConvert2D(S);
downsampling_scale=8;
psf        =    fspecial('gaussian',8, 2);
par.fft_B      =    psf2otf(psf,[M N]);
par.fft_BT     =    conj(par.fft_B);
s0=1;
par.H          =    @(z)H_z(z, par.fft_B, downsampling_scale, [M N],s0 );
par.HT         =    @(y)HT_y(y, par.fft_BT, downsampling_scale,  [M N],s0);
Y_h_bar=par.H(S_bar);

  
SNRh=35;
sigma = sqrt(sum(Y_h_bar(:).^2)/(10^(SNRh/10))/numel(Y_h_bar));
rng(10,'twister')
   Y_h_bar = Y_h_bar+ sigma*randn(size(Y_h_bar));
HSI=hyperConvert3D(Y_h_bar,M/downsampling_scale, N/downsampling_scale );
HSI(HSI<0)=0;



%%  simulate HR-MSI
rng(10,'twister')
Y = F*S_bar;
SNRm=30;
sigmam = sqrt(sum(Y(:).^2)/(10^(SNRm/10))/numel(Y));
Y = Y+ sigmam*randn(size(Y));
MSI=hyperConvert3D(Y,M,N);
MSI(MSI<0)=0;

%% E2E-Fusion
para.p=6;
path_onnx = 'ResNet.onnx';  %ResNet
t0=datetime('now');
lam1 = 1;
[E2E_fusion_washingtondc]= Fusion_E2E( HSI, MSI,F,par.fft_B,downsampling_scale,S,para,lam1,path_onnx);
time_e2e_fusion=datetime('now') - t0;
[MPSNR,PSNRV,MSSIM,SSIMV,MFSIM,FSIMV]=QuanAsse_psnr_ssim_fsim(S,E2E_fusion_washingtondc);