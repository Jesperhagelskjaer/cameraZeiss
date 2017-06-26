function [ y ] = IIRLinearFilter( x )

a = [1.000000000000000 -3.094801825566953 3.355904661293955 -1.868576820251737 1.343811361028703  -0.943260898767957 0.145954174911044 0.035532108010154 0.025478342815136];
b = [0.150087125072278 0 -0.600348500289111 0  0.900522750433667 0 -0.600348500289111 0 0.150087125072278];
y = filtfilt(b,a,x);

end

