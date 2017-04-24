@echo off
set BPATH=%PATH%
set PATH=..\api\lib;%PATH%
if exist updateaxcam.cmd call build

ccam %1 %2 %3 %4 %5

:end
set PATH=%BPATH%
