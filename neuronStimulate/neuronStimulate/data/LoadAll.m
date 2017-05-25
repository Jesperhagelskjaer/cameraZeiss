clear;
close all;
fileList = dir('LX_*.txt');

for i=1:size(fileList,1)
    channelName = fileList(i).name;
    if (channelName(20) == 'D')
        chNumber(i) = str2double(channelName(21:22));
        channel = load(channelName);
        channels(:,i) = channel(1:end);
    end
    if (channelName(20) == 'H')
        header = load(channelName);
    end
end

length = 10000;
figure,
surf(channels(1:length,:));
xlabel('channels');
ylabel('samples');
zlabel('amplitude');

figure
plot(header(1:length,3));
xlabel('samples');
% 0 = stopped
% 1 = measure average
% 2 = search maximum peak
ylabel('mode');
