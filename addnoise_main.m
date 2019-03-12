clc;
clear all;
close all;
tic
%%
rev_path = '/search/speech/cuiguohui/python-conv/pcm_conv/rev_and_noise_snr-30/';
batch_path = '/search/speech/cuiguohui/python-conv/pcm_conv/rev_pickup/';


filelist = get_filename(batch_path,1);
fprintf('file list lentgh is %d \n',length(filelist));

%clean_data =read_pcm( 'clean.pcm');

noise = read_M_pcm('5461.pcm',2);
%size(noise)
parpool('local',16)
parfor fidx =1:length(filelist)
%for fidx =1:1

	data =read_M_pcm(strcat(batch_path,char(filelist(fidx))),2);
	%size(data)
        y_out=add_noise(data,noise,-30);	
        write_M_pcm(y_out,rev_path,char(filelist(fidx)));

end

delete(gcp('nocreate'))









