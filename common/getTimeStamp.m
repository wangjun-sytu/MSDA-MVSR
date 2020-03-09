function ts = getTimeStamp()
% 得到当前的时间戳，形成时间戳字符串
T = clock();
start = 1;
for i = 1: 6
    num = round(T(i));
    str = num2str(num);
    len = length(str);
    if (len<2)
        ts(start: start+len) = ['0' str];
        start = start+len+1;
    else
        ts(start: start+len-1) = str;
        start = start+len;
    end
end

        