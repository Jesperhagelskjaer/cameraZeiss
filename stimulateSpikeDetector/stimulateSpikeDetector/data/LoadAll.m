clear;
close all;
fileList = dir('LX_*.txt');
NUM_CH = 32;
NUM_TEMPLATES = 64;

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
    if (channelName(20) == 'C') % Stimulate Active Channels
        result = load(channelName);
        costs = result(:,1); % First field cost
        activeCh = result(:,2); % Second field active channel
        maximum = result(:,3:NUM_CH+2); % 3-34 Peak value 
        average = result(:,NUM_CH+3:2*NUM_CH+2); % 35-66 Average value
    end
    if (channelName(20) == 'T') % Stimulate Single Neuron with Spike Detector
        result = load(channelName);
        costs = result(:,1); % First field cost
        activeTemplate = result(:,2); % Second field active template
        templateSpikeCounts = result(:,3:NUM_TEMPLATES+2); % 3-66 Neuron spike counts
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
title('Modes: stop(0), average(1), analyse(2)');
ylabel('mode');
xlabel('sample');

ch31 = channels(1:length, 31);
ch32 = channels(1:length, 32);
figure,
plot(ch31);
hold on;
plot(ch32);
title('Channel 32(red) and 31(blue)');
ylabel('amplitude');
xlabel('sample');

idxZero = header(1:length, 3) < 2;
diff = ch32 - ch31;
diff(idxZero) = 0;
figure,
plot(diff);
title('Difference between channel 31 and 32');
ylabel('amplitude');
xlabel('sample');
max(diff)

figure,
plot(costs)
title('Cost');
ylabel('amplitude');
xlabel('sample');
mean(costs)

if exist('templateSpikeCounts', 'var')
    figure, 
    surf(templateSpikeCounts);
    xlabel('template');
    ylabel('iteration');
    zlabel('accumulated counts');
    title('Accumulated Counts of Neuron Spikes');
end
