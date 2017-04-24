@echo off
set BPATH=%PATH%
set PATH=..\QtWinMin\4.8.6\bin;..\api\lib;%PATH%
if  exist ..\QtWinMin goto qt_ok
cd ..
echo Extracting Qt ..
bin\unzip -q QtWinMin
echo done.
cd mcam

:qt_ok
if exist updateaxcam.cmd call build

mcam

:end
set PATH=%BPATH%
