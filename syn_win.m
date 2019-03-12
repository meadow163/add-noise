function win_syn = syn_win(win_ana)
%WIN Summary of this function goes here
%   Detailed explanation goes here
%global nstep;
norm_win=win_ana.*win_ana;
m_block_num=2;m_shift_size=length(win_ana)/2;
for i=1:1:m_shift_size
    temp=0;
    for j=1:1:m_block_num
        temp=temp + norm_win(i+(j-1)*m_shift_size);
    end
    norm_win(i)=1/temp;
end

for i=1:1:m_shift_size
    norm_win(i+m_shift_size)=norm_win(i);
end

win_syn=norm_win.*win_ana;


end

