function ts = getTimeStamp()
% �õ���ǰ��ʱ������γ�ʱ����ַ���
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

        