function lamdat = get_lamda(Y, frm, freq, alpha, pre_idx, micNum)

    if frm < pre_idx+1
        yt = Y(frm-1, :, freq);
        lamdat = 0.05 * (yt*yt');
    else
        yt = Y(frm, :, freq);
        %sprintf('%f,%f',real(yt),imag(yt))
        yt_theta = Y(frm-pre_idx, :, freq);
        lamdat = (yt*yt' - alpha * (yt_theta * yt_theta')) / micNum;
% for limit
    if lamdat < 0
        lamdat = 0.05 * (yt*yt');
    end
    end
end