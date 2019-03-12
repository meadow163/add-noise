function [ z ] = read_M_pcm(path,M)
%READ_PCM Summary of this function goes here
%   Detailed explanation goes here

fid1=fopen(path,'r');
x = fread(fid1,inf,'int16');
y=reshape(x,M,[]);
z=y.';
fclose(fid1);

end

