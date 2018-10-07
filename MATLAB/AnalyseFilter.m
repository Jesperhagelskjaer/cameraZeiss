function [ filter1 ] = AnalyseFilter( filter1 )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%% ###########################################################
%# Filter analysis code
% Calculate the frequencies of interest     

filter1.p_poles = roots( filter1.denominator );
filter1.z_zeros = roots( filter1.numerator );

filter1.frequencies = linspace( 0, 1, 512 );


% Calculate the complex frequency response      
filter1.cmplx = freqz( filter1.numerator, filter1.denominator, filter1.frequencies * pi );


% Calculate the magnitude of the frequency response
filter1.magnitude = abs( filter1.cmplx );


% Calculate the phase response      
filter1.phase = angle( filter1.cmplx );
% filter1.phase = unwrap( arg( filter1.cmplx ) );


% Calculate the group delay      
filter1.delay = grpdelay( filter1.numerator, filter1.denominator, filter1.frequencies );


% Calculate the impulse response      
filter1.impulse = filter( filter1.numerator, filter1.denominator, [1, zeros(1,127)] );


% Calculate the step response      
filter1.step = filter( filter1.numerator, filter1.denominator, ones(1,128) );


%% ###########################################################
%# Generate graphs
% The frequency vs magnitude graph
figure(2)
subplot( 3, 2, 1 );plot( filter1.frequencies, 	filter1.magnitude );
xlabel('Normalized Frequency');
ylabel('Magnitude');
title( 'Magnitude' );
      
% The frequency vs phase graph
subplot( 3, 2, 3 );
plot( filter1.frequencies, 	unwrap( filter1.phase ) );
xlabel('Normalized Frequency');
ylabel('Phase');
title( 'Phase' );


% The frequency vs group delay graph
subplot( 3, 2, 5 );
plot( filter1.frequencies, 	filter1.delay );
title( 'Group Delay (samples)' );
xlabel('Normalized Frequency');
ylabel('Group Delay');
      
% The impulse response vs time graph
subplot( 3, 2, 2 );
stem( filter1.impulse );
title( 'Impulse Response' );
xlabel('Time (Samples)');
ylabel('Impulse Response');
      
subplot( 3, 2, 4 );
stem( filter1.step );     
title( 'Step Response' );
xlabel('Time (Samples)');
ylabel('Step Response');


% z-plane plot (poles and zeros)
subplot( 3, 2, 6 );
zplane( filter1.z_zeros, filter1.p_poles );
title( 'Poles and Zeros' );

%z-plane plot (Coefficients)
%figure
%zplane( filter1.numerator, filter1.denominator );

figure(3)

stem(filter1.numerator);
title('FIR filter coefficients');

figure(4)
freqz(filter1.numerator, filter1.denominator)
title('FIR frequency response');

filter1


end

