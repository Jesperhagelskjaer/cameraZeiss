/*
 * Filter Coefficients (C Source) generated by the Filter Design and Analysis Tool
 * Generated by MATLAB(R) 9.1 and the Signal Processing Toolbox 7.3.
 * Generated on: 19-Jun-2017 18:55:33
 */

/*
 * FDATOOL: Bandpass filter using Window method with Hanning, fc2 = 8 kHz
 * Discrete-Time FIR Filter (real)
 * -------------------------------
 * Filter Structure  : Direct-Form FIR
 * Filter Length     : 257
 * Stable            : Yes
 * Linear Phase      : Yes (Type 1)
 */

const double B300Hz[TAPS] = {
  -4.754706248493e-05,-0.000348328656801,-0.0003225193607791,-2.770151751702e-05,
  -0.000124714558319,-0.0004194385402906,-0.0002637287005877,5.917407596799e-06,
  -0.0002239827980428,-0.0004690270517818,-0.0001762062645508,1.964589151116e-05,
  -0.0003424299324266,-0.0004782625726193,-5.45033568462e-05,4.422822395593e-06,
  -0.0004664011124462,-0.000419478405924, 9.99109506478e-05,-4.864480696347e-05,
  -0.0005668659961704,-0.0002633077464231,0.0002734194479289,-0.0001417138659192,
  -0.0005990257533298,1.039333175826e-05,0.0004391453395862,-0.000262291156145,
  -0.0005080592581056, 0.000400714854052,0.0005614710262993,-0.0003770102727531,
  -0.0002412860257345,0.0008767066181622, 0.000605969654131,-0.0004303993473978,
  0.0002349981454341, 0.001374858913617,0.0005525981905544,-0.0003511716902197,
  0.0009176555465503, 0.001805447137581, 0.000408413674038,-6.683769389863e-05,
   0.001752666719951, 0.002068017663798,0.0002153478147803,0.0004749148388647,
   0.002629855752417,  0.00207374567564,4.904220191445e-05, 0.001282029144104,
   0.003390079328975, 0.001770055121406,6.452066796209e-06, 0.002296839204277,
   0.003846143680608, 0.001161405835003,0.0001825880447787,  0.00338417184994,
   0.003815540021986, 0.000320022236525,0.0006398134727365, 0.004334289009282,
   0.003159871974309,-0.0006182308461639, 0.001375812035953, 0.004883751447888,
   0.001823373271923,-0.001474790844263, 0.002297951734304, 0.004753725183047,
  -0.000138128939944,-0.002064426893164, 0.003211746631151, 0.003701253001026,
  -0.002545641904347,-0.002240857773328, 0.003829246394774, 0.001575470357138,
   -0.00510798054923,-0.001943527681747, 0.003799674236462,-0.001631449139008,
  -0.007456294745184, -0.00123582077125, 0.002760102904288,-0.005748982258531,
  -0.009198332392591,-0.0003258891175219,0.0003993191465171, -0.01041669344958,
  -0.009984115504146,0.0004354865218871, -0.00347569837077,  -0.0150999110995,
  -0.009571532534166,0.0005801295800388,-0.008883168894584, -0.01912934529395,
  -0.007879295146669,-0.000436653849396, -0.01563175926945, -0.02175215365602,
  -0.005016149306402,-0.003197790112111, -0.02331670460829,   -0.022167610815,
  -0.001279030352454,-0.008327130687843, -0.03135077122955, -0.01948806643204,
   0.002881641393079, -0.01662862689109, -0.03902783140798, -0.01245451872252,
   0.006925158947932, -0.02969368183053, -0.04561064576914, 0.001816014590635,
    0.01030225379472,  -0.0527623527374, -0.05042954406469,  0.03531618495991,
    0.01254150334548,  -0.1205683925321, -0.05297608826587,   0.2963586391734,
      0.513022464946,   0.2963586391734, -0.05297608826587,  -0.1205683925321,
    0.01254150334548,  0.03531618495991, -0.05042954406469,  -0.0527623527374,
    0.01030225379472, 0.001816014590635, -0.04561064576914, -0.02969368183053,
   0.006925158947932, -0.01245451872252, -0.03902783140798, -0.01662862689109,
   0.002881641393079, -0.01948806643204, -0.03135077122955,-0.008327130687843,
  -0.001279030352454,   -0.022167610815, -0.02331670460829,-0.003197790112111,
  -0.005016149306402, -0.02175215365602, -0.01563175926945,-0.000436653849396,
  -0.007879295146669, -0.01912934529395,-0.008883168894584,0.0005801295800388,
  -0.009571532534166,  -0.0150999110995, -0.00347569837077,0.0004354865218871,
  -0.009984115504146, -0.01041669344958,0.0003993191465171,-0.0003258891175219,
  -0.009198332392591,-0.005748982258531, 0.002760102904288, -0.00123582077125,
  -0.007456294745184,-0.001631449139008, 0.003799674236462,-0.001943527681747,
   -0.00510798054923, 0.001575470357138, 0.003829246394774,-0.002240857773328,
  -0.002545641904347, 0.003701253001026, 0.003211746631151,-0.002064426893164,
  -0.000138128939944, 0.004753725183047, 0.002297951734304,-0.001474790844263,
   0.001823373271923, 0.004883751447888, 0.001375812035953,-0.0006182308461639,
   0.003159871974309, 0.004334289009282,0.0006398134727365, 0.000320022236525,
   0.003815540021986,  0.00338417184994,0.0001825880447787, 0.001161405835003,
   0.003846143680608, 0.002296839204277,6.452066796209e-06, 0.001770055121406,
   0.003390079328975, 0.001282029144104,4.904220191445e-05,  0.00207374567564,
   0.002629855752417,0.0004749148388647,0.0002153478147803, 0.002068017663798,
   0.001752666719951,-6.683769389863e-05, 0.000408413674038, 0.001805447137581,
  0.0009176555465503,-0.0003511716902197,0.0005525981905544, 0.001374858913617,
  0.0002349981454341,-0.0004303993473978, 0.000605969654131,0.0008767066181622,
  -0.0002412860257345,-0.0003770102727531,0.0005614710262993, 0.000400714854052,
  -0.0005080592581056,-0.000262291156145,0.0004391453395862,1.039333175826e-05,
  -0.0005990257533298,-0.0001417138659192,0.0002734194479289,-0.0002633077464231,
  -0.0005668659961704,-4.864480696347e-05, 9.99109506478e-05,-0.000419478405924,
  -0.0004664011124462,4.422822395593e-06,-5.45033568462e-05,-0.0004782625726193,
  -0.0003424299324266,1.964589151116e-05,-0.0001762062645508,-0.0004690270517818,
  -0.0002239827980428,5.917407596799e-06,-0.0002637287005877,-0.0004194385402906,
  -0.000124714558319,-2.770151751702e-05,-0.0003225193607791,-0.000348328656801,
  -4.754706248493e-05
};
