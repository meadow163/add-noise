function write_M_pcm( y,path,filename)
%READ_PCM Summary of this function goes here
%   Detailed explanation goes here
    z=reshape(y,1,[]);
    des=strcat(path,char(filename));
    fid1=fopen(des,'w');
    x = fwrite(fid1,z,'int16');
    fclose(fid1);


end

