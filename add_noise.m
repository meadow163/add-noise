%clc;
%clear all;
%close all;

function [y] = add_noise(x,noise,SNR)


L=size(x,1)
n_part=noise(1:L,:);

x_power = sum(x.*x);
noise_power=sum(n_part.*n_part);
N = x_power/(10^(SNR/10));
factor = sqrt(N/noise_power)
y=x+factor*noise(1:L,:);
y=y.';
end 
