function [ fileNames ] = get_filename( path ,wav_flag)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
fileFolder=char(fullfile(path));%
if(wav_flag == 1)
    dirOutput=dir(fullfile(fileFolder,'*.pcm'));%������ڲ�ͬ���͵��ļ����á�*����ȡ���У������ȡ�ض������ļ���'.'�����ļ����ͣ������á�.jpg��
else
    dirOutput=dir(fullfile(fileFolder,'*.pcm'));%������ڲ�ͬ���͵��ļ����á�*����ȡ���У������ȡ�ض������ļ���'.'�����ļ����ͣ������á�.jpg��
end
fileNames={dirOutput.name}';

end

