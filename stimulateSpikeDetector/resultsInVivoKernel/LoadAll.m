clear;
close all;

% Results precision weight = 0.9, recall weight = 0.1, predict time = 120 sec
fileList = dir('LX_*085553*.txt'); % Kernel 3x3 laplacian, best score in TRAIN_20180317_08553.txt
%fileList = dir('LX_*100411*.txt'); % Without Kernel 3x3, see score in TRAIN_20180317_100411.txt

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

length = 20000;
figure,
surf(channels(1:length,:));
xlabel('channels');
ylabel('samples');
zlabel('amplitude');

figure
plot(header(1:length*2,3));
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
    iterations = size(templateSpikeCounts,1);

%     for i=1:size(templateSpikeCounts,2)
%         spikesCount = templateSpikeCounts(iterations,i);
%         if (iterations > spikesCount && ...
%             spikesCount > 10)
%             % Plot only valid templates where spikes found 
%             plot(templateSpikeCounts(:,i));
%             name = ['Template #' num2str(i)];
%             title(name);
%             xlabel('iteration');
%             ylabel('accumulated counts');
%             pause(1);
%         end
%     end
    
    name1 = 'Templates (5-100):';
    iterLim1 =  100;
    name2 = 'Templates (100-1000):';
    iterLim2 = 1000;
    name3 = 'Templates (1000-25000):';
    iterLim3 = 25000;
    for i=1:size(templateSpikeCounts,2)
        spikesCount = templateSpikeCounts(iterations,i);
        if (iterLim1 > spikesCount && spikesCount > 5)
            figure(7)
            plot(templateSpikeCounts(:,i));
            hold on
            xlabel('iteration');
            ylabel('accumulated counts');
            name1 = [name1 ' T' num2str(i)];
        end
        if (iterLim2 > spikesCount && spikesCount > iterLim1)
            figure(8)
            plot(templateSpikeCounts(:,i));
            hold on
            xlabel('iteration');
            ylabel('accumulated counts');            
            name2 = [name2 ' T' num2str(i)];
        end
        if (iterLim3 > spikesCount && spikesCount > iterLim2)
            figure(9)
            plot(templateSpikeCounts(:,i));
            hold on
            xlabel('iteration');
            ylabel('accumulated counts');            
            name3 = [name3 ' T' num2str(i)];
        end
    end
    figure(7)
    title(name1);
    figure(8)
    title(name2);
    figure(9)
    title(name3);

    %% Plot used templates
    PrePath = 'C:\neuronSpikeDetector\Matlab\'; % Path to the root of this project
    DiectoryToEvaluate = 'C:\2017-04-21_16-58-45'; % Path to the data, rez file and more
    TemplatesFile = strcat(DiectoryToEvaluate,'\templates.npy');
    addpath(strcat(PrePath,'MatlabFunctions'));
    addpath(strcat(PrePath,'MatlabScripts'));

    MaximumChannelsToUse = 32;
    templateGain = 1;
    %templateGain = 6; % 15.6 db
    pathToNPYMaster = strcat(PrePath, 'MatlabLibs\npy-matlab-master'); % Path to NPY matlab reader project
    ViewFiguresRunning = 'YES';
    ShowFunctionExcTime = 'NO';

    for Y = 1: 64
        if (templateSpikeCounts(end, Y) > 50) %% Plot only templates with more than 50 counts
            templateCurrentlyTesting = Y;
            template = PrepareTemplate( TemplatesFile, templateCurrentlyTesting, [1:MaximumChannelsToUse], ...
                                    templateGain, pathToNPYMaster, ViewFiguresRunning, ShowFunctionExcTime);
        end
    end

end


