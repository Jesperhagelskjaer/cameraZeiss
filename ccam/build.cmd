@echo off
if exist updateaxcam.cmd call updateaxcam

:nounzip
nmake -f Makefile.win %1 %2

:end
