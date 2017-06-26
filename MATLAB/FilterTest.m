clear;
close all;

Fs = 30000;
F1 = 300; % Hz
F2 = 9000; % Hz
n = 4;
t = 1:10000;
N = 257;

Impulse = zeros(1,N);
Impulse(1) = 1;
fpass = [F1 F2];
[b,a] = butter(n,fpass*2/Fs);
y = IIRLinearFilter(Impulse);
%y = filtfilt(b,a,Impulse);
plot(y, '.');
figure, freqz(y);
H = abs(fft(y));
xn = (1:ceil(N/2)).*(Fs/N);
figure; plot(xn, 20*log10(H(1:ceil(N/2))));
xlabel('Frequency (Hz)');
ylabel('Amplitude (db)');

%% Test filter
bf = y;
Fc = 500;
s = sin(2*pi*Fc/Fs*t);
% Creating output signal 

nt = 0.5; %sec endtime
st = 0:1/Fs:nt; %starttime / setup time array
sf = 10; %Hz startfreqz
nf = 20000; %Hz endfreqz

% generate cirp
scirp = chirp(st,sf,nt,nf);
figure, plot(scirp);

% Similar FIR filter - not good
y = filter(bf, 1, scirp); 
figure, plot(y);
Hy = abs(fft(y));
M = ceil(length(Hy)/2);
xn = (1:M).*(Fs/(length(Hy)));
figure, plot(xn, 20*log10(Hy(1:M)));

% Refence kilosort butterworth filter
y = filtfilt(b,a,scirp);
figure, plot(y);
Hy = abs(fft(y));
M = ceil(length(Hy)/2);
xn = (1:M).*(Fs/(length(Hy)));
figure, plot(xn, 20*log10(Hy(1:M)));

% FIR filter, 300Hz
Hd = FirFilter300Hz;
y = filter(Hd.Numerator, 1, scirp); 
figure, plot(y);
Hy = abs(fft(y));
M = ceil(length(Hy)/2);
xn = (1:M).*(Fs/(length(Hy)));
figure, plot(xn, 20*log10(Hy(1:M)));


%% Save coefficeints to header file
SaveFilterHeaderFile(bf, F1, 'FilterCoeffs_B300Hz.h');
