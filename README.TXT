************************************************************************
*                                                                      *
*                          Axiocam USB 3.0 SDK                         *
*                                                                      *
*                       Carl Zeiss Microscopy GmbH                     *
*                                                                      *
************************************************************************



************************************************************************
* General *
***********

The Axiocam Software Development Kit contains drivers for the Carl Zeiss 
Axiocam and the related API documentation. 

The SDK supports both 64 bit Linux and 64 bit Windows.

A sample GUI application "mcam" written in C++ based on the Qt-GUI Toolkit
illustrates the basic camera functionality. 
It runs under Linux and windows as well.

Furthermore a simple command line application "ccam" is available which  
can grab an image and store it as a 'tif' file.

*********************************************************
Compilation for was tested with the following tool chain:
*********************************************************

===========================
Linux distributions:

Open SUSE 13.1 64 bit
Ubuntu 14.02.2 64 bit
Fedora 20.1 64 bit
===========================

Be sure to install: C++, Qt4 and "libusb" development packages which comes
with your distribution! 

Use Eclipse (SuSE Only tested with "Eclipse-Luna")
   Import -> Existing Projects into Workspace -> navigate to "mcam" folder
   Indexer:  Project Properties -> C++ General -> Indexer -> Check all options
   Debug: Project Properties -> C++ Build -> Environment -> Edit MCAMDEBUG=1
   Clean -> Build -> Run
   Eclipse should show "0 Problems"
OR Command line 
- Open a shell
- change to folder "AxiocamSDK/mcam"
- For Build: make
- Run with ./mcam

================
Windows 7 64 bit
================
A) Visual Studio Express 2013 Update 4
- Open Visual Studio Project "mcam" in the mcam folder, build and run 
OR
- Open "VS2013 x64 Cross Tools-Command Prompt"
- change to folder "AxiocamSDK\mcam"
- For Build: build.cmd 
- Run with start_mcam

B) Visual Studio Express 2010 SP1 + Windows SDK 7.1 (x64)
- Open x64 "Windows SDK 7.1 Command Prompt"
- change to folder "AxiocamSDK\mcam"
- For Build: build.cmd 
- Run with start_mcam

-------------------------------------------------
For a History of changes see doc/Releasenotes.pdf
-------------------------------------------------

************************************************************************
* Contents *
************

This distribution consists of the following directories:

------------+-----------------------------------------------------------
doc         | API documentation for the camera driver. Available in 
            | Html format.
            |
------------+-----------------------------------------------------------
api/inlude  | C++ header files containing the required data types and 
            | function definitions.
            |
------------+-----------------------------------------------------------
api/lib     | Shared libraries for Linux, DLL and object libraries for 
            | Windows for all Axiocam USB 3.0 models:
            | Currently supported models:
            | Axiocam 503 and 506 color and mono
------------+-----------------------------------------------------------
install     | Windows and Linux drives including installer
            | 
------------+-----------------------------------------------------------
QtWinMin    | Minimum set of a 64 bit Open Source Qt-Toolkit for Windows 
            | including 64 bit binaries to be able to compile "mcam"
            | (See also hint below) 
            | For Linux Qt comes with your 64 bit distribution.
------------+-----------------------------------------------------------
mcam        | Sample application written in C++, demonstrating the 
            | basic camera functionality (Binaries included)
            | 
            | Start mcam with Linux:   ./mcam
            | Start mcam with Windows: start_mcam
            | 
            | Build mcam with Linux:  make
            | Build mcam with Windows build
------------+-----------------------------------------------------------
ccam        | Simple command line application written in C,
            | which is able to grab an image and store it as a 'tif' file.
            | 
            | Start ccam with Linux:   ./ccam
            | Start ccam with Windows: start_ccam
            | 
            | Build ccam with Linux:   make
            | Build ccam with Windows: build
------------+-----------------------------------------------------------


Hint for Windows Users: 
=======================

You may keep using the included Qt Minimal Version (QtWinMin\4.8.6) to compile mcam.

For full development you should download and compile a full Qt version if you plan to develop with Qt for Windows:
E.g. see:
  https://download.qt.io/archive/qt/4.8/4.8.6/
  http://sourceforge.net/projects/qtx64/files/qt-x64/4.8.6/msvc2010/qt-4.8.6-x64-msvc2010-rev1.exe/download

************************************************************************
Copyright 2015 Carl Zeiss Microscopy GmbH. All rights reserved.
************************************************************************
