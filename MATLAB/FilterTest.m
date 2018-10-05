clear;
close all;

Fs = 30000;
F1 = 300; % Hz
F2 = 8000; % Hz
n = 3; % Order of butterworth filter
t = 1:10000;
fpass = [F1 F2];
[b,a] = butter(n,fpass*2/Fs);


% N = 257;
% Impulse = zeros(1,N);
% Impulse(1) = 1;
% y = IIRLinearFilter(Impulse);
% %y = filtfilt(b,a,Impulse);
% plot(y, '.');
% figure, freqz(y);
% H = abs(fft(y));
% xn = (1:ceil(N/2)).*(Fs/N);
% figure; plot(xn, 20*log10(H(1:ceil(N/2))));
% xlabel('Frequency (Hz)');
% ylabel('Amplitude (db)');

%% Test filter
%bf = y;

Fc = 500;
%s = sin(2*pi*Fc/Fs*t);
% Creating output signal 

nt = 0.5; %sec endtime
st = 0:1/Fs:nt; %starttime / setup time array
sf = 10; %Hz startfreqz
nf = 20000; %Hz endfreqz

% generate cirp
scirp = chirp(st,sf,nt,nf);
%figure, plot(scirp);

% Similar FIR filter - not good
y = filter(b, a, scirp); 
y = filter(b, a, y); 
figure, plot(y);
title('Butterworth filtered');
M = ceil(length(y)/2);
Hy = abs(fft(y))/M;
xn = (1:M).*(Fs/(length(Hy)));
figure, plot(xn, 20*log10(Hy(1:M)));
title('Butterworth frequency');
ylabel('db');
xlabel('Hz');

% Refence kilosort butterworth filter
y = filtfilt(b,a,scirp); % Zero phase filter
figure, plot(y);
title('Kilosort butterworth filtered');
Hy = abs(fft(y))/M;
M = ceil(length(Hy)/2);
xn = (1:M).*(Fs/(length(Hy)));
figure, plot(xn, 20*log10(Hy(1:M)));
title('Kilosort butterworth frequency');
ylabel('db');
xlabel('Hz');

% FIR filter, 300Hz
Hd = FirFilter300Hz;
y = filter(Hd.Numerator, 1, scirp); 
figure, plot(y);
title('FIR filtered');
Hy = abs(fft(y))/M;
M = ceil(length(Hy)/2);
xn = (1:M).*(Fs/(length(Hy)));
figure, plot(xn, 20*log10(Hy(1:M)));
title('FIR filter, Window method');
ylabel('db');
xlabel('Hz');

%% Save coefficeints to header file
bf = Hd.Numerator
SaveFilterHeaderFile(bf, F1, 'FilterCoeffs_B300Hz.h');
figure, stem(bf)
