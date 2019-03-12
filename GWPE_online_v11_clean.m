clc;
clear all;
close all;
tic
%%
M=2;
win_size = 512;
win_shift = 256;
block_num = win_size/win_shift;
ana_win=win(win_size);
win_syn=syn_win(ana_win);
fs = 16000;
sp_size = win_size/2+1;
block_num = win_size/win_shift;
%% data preparation
wav_flag = 0;
read_path1= '111.pcm';
read_path2= '222.pcm';
read_path3= '111_delay.pcm';
read_path4= '222_delay.pcm';
wav_flag = 0;% 1:wav other:pcm
% data1=read_pcm( read_path1,wav_flag);
% data2=read_pcm( read_path2,wav_flag);
% data3=read_pcm( read_path3,wav_flag);
% data4=read_pcm( read_path4,wav_flag);
% data= [data1 data2 data3(1:length(data1)) data4(1:length(data1))];
%
% %[data fs]= wavread('reverberant.wav');
% %[data fs]= wavread('meiti_03.wav');
% %data =read_pcm( 'clean.pcm',wav_flag);

%path = 'revdata.pcm';
%data1 =read_pcm( 'D:\matlabproject\NTT_reverb\input\revdata1.pcm');
%data2 =read_pcm( 'D:\matlabproject\NTT_reverb\input\revdata2.pcm');
%data= [data1 data2];

%data =read_M_pcm( 'D:\matlabproject\RIR-Generator-master\gdata\Room_1_10_2_dist_10_degree_-15_beta_0.2.pcm',2);
%data =read_M_pcm( 'D:\matlabproject\NTT_reverb\input\qs1hour_cut.pcm',2);
%data =read_M_pcm( 'dist_2_degree_75_beta_0.6.pcm',2);

%data =read_M_pcm( 'D:\matlabproject\NTT_reverb\input\qs1hour_cut.pcm',2);

% path = 'C:\Users\cuiguohui\Desktop\bojun\sogou20160406_speech_170_14422883.pcm';
% data = read_M_pcm(path,2);
batch_path = '/search/speech/cuiguohui/python-conv/pcm_conv/rev_pickup/';
%dev_path = '/search/speech/cuiguohui/python-conv/pcm_conv/dev_rev_pickup_L15_it1/';
dev_path = '/search/speech/cuiguohui/python-conv/pcm_conv/clean/dev_rev_pickup_L15_it1_fix_0.99/';


%batch_path = '/search/speech/cuiguohui/python-conv/pcm_conv/noise/rev_and_noise_snr0/';
%dev_path =   '/search/speech/cuiguohui/python-conv/pcm_conv/noise/dev_and_noise_snr0_it3/';



filelist = get_filename(batch_path,1);
fprintf('file list lentgh is %d \n',length(filelist));

clean_data =read_pcm( 'clean.pcm');

parpool('local',16)
parfor fidx =1:length(filelist)

data =read_M_pcm(strcat(batch_path,char(filelist(fidx))),2);
len = length( data );
fnum = floor(len/win_shift);
dd=data.';
nfft=win_size;
win=ana_win;
nol = fix(2*nfft/4);
[nmic, nn] = size(dd);

%K = 20;
Nframe =  floor(size(data,1)/win_shift);
%Nframe = min(Nframe,floor(size(clean_data,1)/win_shift));
nfreq = win_size;
% S=zeros(M,Nframe,nfreq);
%==========================================================================
%   7. weighted recursive least square
%==========================================================================
T60 = 0.5;
sigma = 3/(T60*log10(exp(1)));
alphak = 3*log(10)/(0.5*fs);
theta = 50/1000;
psd_size = nfft/2+1;
alpha=  exp(-2*sigma*theta);
tsft = win_shift / fs;
ThetaIdx = fix(theta / tsft);
x_real = zeros(M,psd_size);
var = zeros(1,nfft/2+1);
beta=0.4;%%0.4ื๎บร ,pesq = 2.32
betaW=0.99
array_it = {1};
sz=size(array_it,2);
record_dev = zeros(sz,Nframe);
record_abs = zeros(sz,Nframe);
delta_Tw = zeros(1,Nframe);
gamma_L = zeros(1,Nframe);
gamma_L0 = 0.998;
sfactor = 0.001;
back_len=3;
lambda_t=zeros(Nframe,nfft);
thre = 1.5;
N0=35;
deltabuf=zeros(1,N0);
gamma_s=0.99;
gamma=gamma_s;
stable_win = 375;
T0 = ceil(stable_win/win_shift);
count_f=0;
%%fidn=fopen('C:\Users\cuiguohui\Desktop\bojun\matlab_debug.txt','a+');

record_abs = zeros(257,Nframe);
record_rea = zeros(1,257);
reset_count=0;

sz=1;

for  nn=1:sz
    whole_ite = array_it{nn};
    fd_idx=1;
    delta_tri = 1;
    %order = delta_tri+K-1;
    %gamma = 0.95;%% the best
    % subband order
    K_l=15;
    K = K_l;
    % var init
    Gl = zeros(K*M,M,nfft) ;
    Gl_old = zeros(K*M,M,nfft);
    Y = zeros(K , M,psd_size);
    Y_D_all = zeros(K * M,psd_size);% all oberseved signal of k order with D delay
    x_real = zeros(M,psd_size);
    var = zeros(1,nfft/2+1);
    phyl = zeros(K*M,K*M,nfft);
    for k=1:nfft
        phyl(:,:,k) = 0.01*eye(K*M,K*M);
    end
    % subband index
    DRR=zeros(Nframe,nfft/2+1);
    post_DRR=zeros(Nframe,nfft/2+1);
    % out buffer
    S=zeros(M,Nframe,nfreq);
    %     dreal1=zeros(1,Nframe*nfft);
    %     dreal2=zeros(1,Nframe*nfft);
    %     dima3=zeros(1,Nframe*nfft);
    %     dima4=zeros(1,Nframe*nfft);
    f_buf = [];
    
    kend  = 257;
    %% begin to process
    while(fd_idx < Nframe )
        fd_idx
        delta_wk = 0;
        startidx=(fd_idx-1)*nol+1;
        datBuf = data(startidx:(startidx+nfft-1),1:M);
        y=fft((datBuf) .* (win.' * ones(1,M)));
        %         if fd_idx == 1
        %             for k=1:kend
        %                 var(k) =norm(y(k,:),2)^2;
        %             end
        %         end
        for nn=1:kend
            %overflow:reset and break
            if record_rea(nn) > 10^20
                reset_count=reset_count+1;
                %gamma=0.998;         
                Gl = zeros(K*M,M,nfft) ;
                Gl_old = zeros(K*M,M,nfft);
                phyl = zeros(K*M,K*M,nfft);
                for k=1:nfft
                    phyl(:,:,k) = 0.01*eye(K*M,K*M);
                end
                %record_abs = zeros(257,Nframe);
                break;
            end
        end
        
        for k = 1:kend
            % prepare data
            Y(:,:,k) = [Y(2:end,:,k) ; y(k,:)];
            Y_D_all(:,k) = [Y_D_all(M+1:end, k); Y(max(K-delta_tri,1),:,k).'];
            %Y_D_all(1:K_l*M,k) = [Y(max(K-delta_tri,1),:,k).';Y_D_all(1:(K_l*M-M),k);];
            Yl = Y_D_all(1:K_l*M,k);
            if mod(fd_idx,1) == 0
                lambda_t(fd_idx,k) = get_lamda(Y,max(K, ThetaIdx),k,alpha,ThetaIdx,M);
                %                 if fd_idx > 1
                %                     tmp_d = y(k,:).' - Gl(1:K_l*M,:,k)' * Yl;
                %                     var(k) = beta*var(k) + (1-beta)*max(norm(tmp_d,2)^2/2,0.0001);
                %                 end
                %Gl(1:K_l*M,:,1)
                for it=1:whole_ite
                    %gamma=1;
                    x_pred = y(k,:).' - Gl(1:K_l*M,:,k)'* Yl;
                    tmp = gamma*lambda_t(fd_idx,k) + Yl'*phyl(1:K_l*M,1:K_l*M,k)*Yl;
                    %tmp = gamma*var(k)+Yl'*phyl(1:K_l*M,1:K_l*M,k)*Yl;
                    gain = phyl(1:K_l*M,1:K_l*M,k)*Yl/(abs(real(tmp)));
                    Gl(1:K_l*M,:,k) = Gl(1:K_l*M,:,k) + gain*x_pred';
                    if mod(fd_idx,1) == 0
                        phyl(1:K_l*M,1:K_l*M,k) = (1/gamma)*(phyl(1:K_l*M,1:K_l*M,k)-gain*Yl'*phyl(1:K_l*M,1:K_l*M,k));
                    end
                end
            end
            %output
            rev = Gl(1:K_l*M,:,k)' * Yl;
            x_real(:,k) = y(k,:).' - rev;
            
            %record_abs(k,fd_idx) = norm(Gl(:,k));
            record_rea(k) = norm(Gl(:,k));
            %
            %rou = sqrt(lambda_t(fd_idx,k));
            delta_wk = delta_wk + norm(Gl(1:K_l*M,:,k)-Gl_old(1:K_l*M,:,k))^2/(norm(Gl(1:K_l*M,:,k))^2+eps);
        end
        if(fd_idx>1)
            delta_Tw(fd_idx) = betaW*delta_Tw(fd_idx-1) + (1-betaW)*delta_wk;
            gamma_L(fd_idx) = 2-exp(sfactor*delta_Tw(fd_idx));
            deltabuf = [deltabuf(2:end) delta_Tw(fd_idx)];
            ratio = delta_Tw(fd_idx)/(min(deltabuf)+eps);
            if (ratio < thre || ratio == thre)&&(count_f <= T0)
                %gamma = max(gamma_L0,gamma_L(fd_idx));
                gamma = gamma_s;
                count_f=count_f+1;
            elseif (ratio < thre || ratio == thre)&&(count_f > T0)
                gamma = max(gamma_s,gamma_L(fd_idx));
                gamma = min(gamma,gamma_L0);
                count_f=count_f+1;
            elseif ratio > thre
                gamma = gamma_s;
                count_f=0;
            end
            gamma=0.99;
            
        end
        %Gl(1:K_l*M,:,1)
        %% output
        for k =1:kend
            for j=1:M
                if(abs(x_real(j,k)) > abs( y(k,j)))
                    x_real(j,k) = x_real(j,k) * abs(y(k,j)) /  abs(x_real(j,k));
                end
            end
            S(:,fd_idx,k) =  x_real(:,k);
        end
        %S(:,fd_idx,258:end)=conj(S(:,fd_idx,256:-1:2));
        S(:,fd_idx,(nfft/2 + 2):end)=conj(S(:,fd_idx,nfft/2:-1:2));
        Gl_old = Gl;
        fd_idx=fd_idx+1;
    end
end

% Re-synthesize the obtained source signal
y_out=zeros(nmic,nn);
last_frame=zeros(nfft,1);
tmp_ifft=zeros(nfft,1);
curr_frame=zeros(nfft,1);
nstep=win_shift;
for l=1:M
    for i=1:Nframe
        %i
        tmp_ifft=real(squeeze(ifft(S(l,i,:))));
        %curr_frame=(tmp_ifft)*32768.*win_syn.';
        curr_frame=(tmp_ifft).*win_syn.';
        startidx=(i-1)*nstep+1;
        %gggg=curr_frame(1:nstep) + last_frame(nstep+1:end);
        y_out(l,startidx:startidx+nstep-1)=curr_frame(1:nstep) + last_frame(nstep+1:end);
        last_frame=curr_frame;
    end
end
%data =read_M_pcm(strcat(batch_path,char(filelist(fidx))),2);
write_M_pcm(y_out,dev_path,char(filelist(fidx)));

end
%fclose(fidn);
% fprintf('file finished!\n');
% out_path1= 'D:\matlabproject\two_channel\output\';
% out_path2= 'D:\matlabproject\two_channel\output\';
% write_M_pcm(y_out*32768,'','matlab_out.pcm')
% wavwrite(y_out.',fs,'reverberant_out1.wav');
%savename = numstr()
%write_pcm(y_out(1,:),''./out','sogou20160406_speech_170_14422883.pcm',wav_flag);
%write_pcm(y_out(2,:),'','sogou20160406_speech_170_14422883.pcm',wav_flag);
%[pathstr,name,suffix]= fileparts(path);
%allname = [name ,'_k' ,num2str(K), '_gamma',num2str(gamma), '_it' , num2str(whole_ite), suffix];
%write_M_pcm(y_out,'./out/','dist_2_degree_75_beta_0.6_it3_gama0.99.pcm');
%write_pcm(y_out(1,:)*32768,'./out/','revedata1_ada_out.pcm');
%pesq score
%system('single_wpesq clean.pcm ./out/revedata1.pcm +16000')
delete(gcp('nocreate'))
fprintf('process finished!\n');














