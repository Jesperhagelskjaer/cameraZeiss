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
    if (channelName(20) == 'C')
        costs = load(channelName);
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

ch31 = channels(1:length, 31);
ch32 = channels(1:length, 32);
figure,
plot(ch31);
hold on;
plot(ch32);

idxZero = header(1:length, 3) < 2;
diff = ch32 - ch31;
diff(idxZero) = 0;
figure,
plot(diff);
max(diff)

figure,
plot(costs)