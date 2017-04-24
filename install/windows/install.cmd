@echo off
net session >nul 2>&1
if %errorlevel% equ 0 (
  echo.
) else (
  echo.
  echo ERROR: Administrative rights required! Please run this script as Administrator!
  goto end
)


echo Axiocam NG Driver Install
echo -------------------------
echo.
echo.
echo --------------------------------------------------------------------
echo Press return to start installation of Axiocam USB 3.0 camera drivers
echo --------------------------------------------------------------------
echo.
pause

setlocal enableextensions
cd /d "%~dp0"

rem uninstall ftdi terminal driver if installed

echo - Remove Terminal Driver
cd terminal_uninstall
call uninstall
cd ..

echo - Install Axiocam Power Driver
cd winpwrdrv
dpinst
cd ..

echo - Install Axiocam USB 3.0 Driver
cd usb3
dpinst 
cd ..

echo.
echo done.
echo.
:end
pause
