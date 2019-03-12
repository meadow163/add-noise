function [ x ] = read_pcm( path)
%READ_PCM Summary of this function goes here
%   Detailed explanation goes here
if 1
    %[x,fs,nbit] = wavread(path);
%else
    fid1=fopen(path,'r');
    x = fread(fid1,inf,'int16');
    fclose(fid1);
end



end

