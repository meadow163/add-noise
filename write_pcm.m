function write_pcm( y,path,filename)
%READ_PCM Summary of this function goes here
%   Detailed explanation goes here
if 0%wav_flag ==1
    
    des=strcat(path,char(filename));
    wavwrite(y,16000,des);
else
    des=strcat(path,char(filename));
    fid1=fopen(des,'w');
    x = fwrite(fid1,y,'int16');
    fclose(fid1);
end

end

