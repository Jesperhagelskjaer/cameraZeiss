/*
 * Filter Coefficients (C Source) generated by the Filter Design and Analysis Tool
 * Generated by MATLAB(R) 9.1 and the Signal Processing Toolbox 7.3.
 * Generated on: 19-Jun-2017 18:54:59
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

const double B100Hz[TAPS] = {
  5.940516272983e-05,-0.000242398296786,-0.0002177205635686,7.575147451808e-05,
  -2.349486134739e-05,-0.0003212424353992,-0.0001688473589251,9.677227481485e-05,
  -0.0001385561425326,-0.0003901266022033,-0.0001045775127903,8.250520561367e-05,
  -0.000290443084615,-0.000438724479179,-2.888824182221e-05,1.357822040394e-05,
  -0.0004766146707116,-0.0004512349289975,4.416470106926e-05,-0.0001321084831301,
  -0.0006817342738171,-0.0004124047325513,8.650193984649e-05,-0.0003713525279024,
  -0.0008756616238385,-0.0003166965219665,5.681372717075e-05,-0.0007056929223597,
  -0.001017051919626,-0.0001778747845405,-9.257243661807e-05,-0.001112763479253,
  -0.001063055876113,-3.548758375137e-05,-0.0004032177310033,-0.001542524349741,
  -0.0009836534162894,4.509191499448e-05,-0.0008949439928215,-0.001921212825532,
  -0.0007772871990106,-1.885239540898e-05,-0.001551110196621,-0.002164128039429,
  -0.0004832478887456,-0.0003104025849893,-0.002309560916867,-0.002195983691717,
  -0.0001861433173508,-0.0008900677845242,-0.003063493126527, -0.00197509433029,
  -8.955407723661e-06, -0.00177168108872,-0.003674695245293,-0.001515761759748,
  -9.352715023976e-05, -0.00290331244272,-0.003998839289625,-0.000902485375043,
  -0.0005703811035459,-0.004159388181957,-0.003919333605944,-0.0002903905101479,
  -0.001522866416208,-0.005348651117775,-0.003383420599513,0.0001114339411195,
  -0.002952990046424,-0.006239648314639, -0.00243248022842,7.353988078981e-05,
   -0.00475723169981,-0.006601639434123,-0.001218438246688,-0.0006074064147885,
  -0.006719763321729,-0.006254987594241,-3.014890757917e-18,-0.002056859279945,
  -0.008527778220966,-0.005122139287321, 0.000884066192717, -0.00427511279849,
  -0.009809498089884,-0.003269034514581, 0.001062801505077,-0.007104475183189,
   -0.01019065265774,-0.0009277510398641, 0.000195568261395, -0.01021870712683,
  -0.009360815985538, 0.001505458519626,-0.001961160287446, -0.01313802743841,
  -0.007137937054002, 0.003496981988879,-0.005495143494297, -0.01526634252531,
  -0.003518497303367, 0.004420025104698, -0.01029992811351, -0.01593830909635,
   0.001297671817109, 0.003597703304516, -0.01606586346717,  -0.0144496273935,
   0.006914926462358,0.0003062926846894, -0.02230492005669, -0.01001159392086,
    0.01278308146329,-0.006357404819528, -0.02840755547139,-0.001459581124566,
    0.01826752982582, -0.01807711279218,  -0.0337237588672,   0.0140074855705,
    0.02273767576733, -0.04017442323383, -0.03765551386136,  0.04832578884836,
    0.02565927292525,  -0.1074720036863, -0.03974527746294,   0.3099538531777,
     0.5268312252127,   0.3099538531777, -0.03974527746294,  -0.1074720036863,
    0.02565927292525,  0.04832578884836, -0.03765551386136, -0.04017442323383,
    0.02273767576733,   0.0140074855705,  -0.0337237588672, -0.01807711279218,
    0.01826752982582,-0.001459581124566, -0.02840755547139,-0.006357404819528,
    0.01278308146329, -0.01001159392086, -0.02230492005669,0.0003062926846894,
   0.006914926462358,  -0.0144496273935, -0.01606586346717, 0.003597703304516,
   0.001297671817109, -0.01593830909635, -0.01029992811351, 0.004420025104698,
  -0.003518497303367, -0.01526634252531,-0.005495143494297, 0.003496981988879,
  -0.007137937054002, -0.01313802743841,-0.001961160287446, 0.001505458519626,
  -0.009360815985538, -0.01021870712683, 0.000195568261395,-0.0009277510398641,
   -0.01019065265774,-0.007104475183189, 0.001062801505077,-0.003269034514581,
  -0.009809498089884, -0.00427511279849, 0.000884066192717,-0.005122139287321,
  -0.008527778220966,-0.002056859279945,-3.014890757917e-18,-0.006254987594241,
  -0.006719763321729,-0.0006074064147885,-0.001218438246688,-0.006601639434123,
   -0.00475723169981,7.353988078981e-05, -0.00243248022842,-0.006239648314639,
  -0.002952990046424,0.0001114339411195,-0.003383420599513,-0.005348651117775,
  -0.001522866416208,-0.0002903905101479,-0.003919333605944,-0.004159388181957,
  -0.0005703811035459,-0.000902485375043,-0.003998839289625, -0.00290331244272,
  -9.352715023976e-05,-0.001515761759748,-0.003674695245293, -0.00177168108872,
  -8.955407723661e-06, -0.00197509433029,-0.003063493126527,-0.0008900677845242,
  -0.0001861433173508,-0.002195983691717,-0.002309560916867,-0.0003104025849893,
  -0.0004832478887456,-0.002164128039429,-0.001551110196621,-1.885239540898e-05,
  -0.0007772871990106,-0.001921212825532,-0.0008949439928215,4.509191499448e-05,
  -0.0009836534162894,-0.001542524349741,-0.0004032177310033,-3.548758375137e-05,
  -0.001063055876113,-0.001112763479253,-9.257243661807e-05,-0.0001778747845405,
  -0.001017051919626,-0.0007056929223597,5.681372717075e-05,-0.0003166965219665,
  -0.0008756616238385,-0.0003713525279024,8.650193984649e-05,-0.0004124047325513,
  -0.0006817342738171,-0.0001321084831301,4.416470106926e-05,-0.0004512349289975,
  -0.0004766146707116,1.357822040394e-05,-2.888824182221e-05,-0.000438724479179,
  -0.000290443084615,8.250520561367e-05,-0.0001045775127903,-0.0003901266022033,
  -0.0001385561425326,9.677227481485e-05,-0.0001688473589251,-0.0003212424353992,
  -2.349486134739e-05,7.575147451808e-05,-0.0002177205635686,-0.000242398296786,
  5.940516272983e-05
};
