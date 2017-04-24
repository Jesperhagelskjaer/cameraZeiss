@echo off
if exist updateaxcam.cmd call updateaxcam
if exist updatewinwrappers.cmd call updatewinwrappers

if not exist ..\QtWinMin.zip goto nounzip
nmake -f Makefile.win %1
goto end

:nounzip
nmake -f Makefile.win mcam.exe %1

:end
