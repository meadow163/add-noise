function win_ana=win( win_len )
%WIN Summary of this function goes here
%   Detailed explanation goes here
Nwin_=1:1:win_len;
win_ana=0.54-0.46*cos(2*pi*Nwin_/(win_len-1));

end

