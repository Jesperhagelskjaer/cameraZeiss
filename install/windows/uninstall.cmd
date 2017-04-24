@echo off
net session >nul 2>&1
if %errorlevel% equ 0 (
  echo.
) else (
  echo.
  echo ERROR: Administrative rights required! Please run this script as Administrator!
  goto end
)


echo Axiocam NG Driver Uninstall
echo ---------------------------
echo.
echo.
echo -----------------------------------------------------------------------
echo Press return to start un-installation of Axiocam USB 3.0 camera drivers
echo -----------------------------------------------------------------------
echo.
pause

setlocal enableextensions
cd /d "%~dp0"

echo - Remove Terminal Driver
cd terminal_uninstall
call uninstall
cd ..

echo - Uninstall Axiocam Power Driver
cd winpwrdrv
DPInst.exe /D /U axiocampwr.inf

cd ..

echo - Uninstall Axiocam USB 3.0 Driver

cd usb3
DPInst.exe /D /U axiocam-usb3.inf

echo.
echo done.
echo.
:end
pause
