function [ fileNames ] = get_filename( path ,wav_flag)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
fileFolder=char(fullfile(path));%
if(wav_flag == 1)
    dirOutput=dir(fullfile(fileFolder,'*.pcm'));%如果存在不同类型的文件，用‘*’读取所有，如果读取特定类型文件，'.'加上文件类型，例如用‘.jpg’
else
    dirOutput=dir(fullfile(fileFolder,'*.pcm'));%如果存在不同类型的文件，用‘*’读取所有，如果读取特定类型文件，'.'加上文件类型，例如用‘.jpg’
end
fileNames={dirOutput.name}';

end

