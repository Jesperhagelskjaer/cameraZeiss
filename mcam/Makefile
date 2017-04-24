all: axcam mcam

MAKEFLAGS=-j4

ifeq ($(MCAMDEBUG),1)
  # debug
  CXX_DEBUG_FLAGS=-g
  LINK_DEBUG_FLAGS=-g
  DPATH=Debug
else
  #release
  CXX_DEBUG_FLAGS=-O2
  LINK_DEBUG_FLAGS=-O3
  DPATH=Release
endif

# find Qt4 include folders
QTINCLUDE=/usr/include/QtGui
QTINCLUDE_UBUNTU=/usr/include/qt4/QtGui
ifneq ("$(wildcard $(QTINCLUDE))","")
    QTINCLUDE = -I/usr/include/QtGui -I/usr/include/QtCore
else 
    QTINCLUDE = -I/usr/include/qt4 -I/usr/include/qt4/QtGui -I/usr/include/qt4/QtCore
endif   

# find qt4 executables
QTBINPATH=/usr/bin
UIC=$(QTBINPATH)/uic
MOC=$(QTBINPATH)/moc
ifeq ("$(wildcard $(UIC))","")
    QTBINPATH=/usr/lib64/qt4/bin
    UIC=$(QTBINPATH)/uic
    MOC=$(QTBINPATH)/moc
    ifeq ("$(wildcard $(UIC))","")
        $(warning #### Warning Qt4 uic not found. in $(QTBINPATH))
        UIC=uic
        MOC=moc
    endif
endif

HFILES=include/MCam.hpp include/Application.hpp include/EditProperties.hpp include/ui_mcam.h

axcam:
	   cd ../../fw/cam_usb/axcam && $(MAKE) MAKEFLAGS=
.PHONY: axcam

# qt ubuntu 14.04 /usr/include/qt4/QtGui
# qt suse 13.1 /usr/include
CXX_FLAGS = -lpthread -std=c++0x $(CXX_DEBUG_FLAGS) -DNDEBUG  -I/usr/include/libusb-1.0 $(QTINCLUDE) -Iinclude -I../api/include
CXX_DEFINES = -DQT_CORE_LIB -DQT_GUI_LIB -DQT_NO_DEBUG

mcam: Main.o Application.o Application_moc.o MCam.o Application_moc.o MCamCameraIF_moc.o MCamImage_moc.o ConfigReader.o EditProperties.o EditProperties_moc.o MCamCameraIF.o MCamUtil.o MCamImage.o MCamTest.o mcamlogger.o
	   c++  -std=c++0x $(LINK_DEBUG_FLAGS) -DNDEBUG  Main.o Application.o MCam.o Application_moc.o MCamCameraIF_moc.o MCamImage_moc.o ConfigReader.o EditProperties.o EditProperties_moc.o\
        MCamCameraIF.o MCamUtil.o MCamImage.o MCamTest.o mcamlogger.o -o mcam -rdynamic -lQtGui -lQtCore -lrt -ldl \
        -lQtGui -lQtCore -lrt -ldl -lusb-1.0 -lpthread ../api/lib/libaxcam64.so -Wl,-rpath,../api/lib

include/ui_mcam.h: mcam.ui
	   $(UIC) -o include/ui_mcam.h mcam.ui

src/Application_moc.cpp: src/Application.cpp include/ui_mcam.h $(HFILES)
	   $(MOC) -o src/Application_moc.cpp include/Application.hpp

src/MCamCameraIF_moc.cpp: src/MCamCameraIF.cpp include/ui_mcam.h $(HFILES)
	   $(MOC) -o src/MCamCameraIF_moc.cpp include/MCamCameraIF.hpp

src/MCamImage_moc.cpp: src/MCamImage.cpp include/ui_mcam.h $(HFILES)
	   $(MOC) -o src/MCamImage_moc.cpp include/MCamImage.hpp

src/EditProperties_moc.cpp: src/EditProperties.cpp include/ui_mcam.h $(HFILES)
	   $(QTBINPATH)/moc -o src/EditProperties_moc.cpp include/EditProperties.hpp

Application.o: src/Application.cpp  include/ui_mcam.h $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/Application.cpp

Application_moc.o: src/Application_moc.cpp $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/Application_moc.cpp

MCamCameraIF_moc.o: src/MCamCameraIF_moc.cpp $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/MCamCameraIF_moc.cpp

MCamImage_moc.o: src/MCamImage_moc.cpp $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/MCamImage_moc.cpp

EditProperties.o: src/EditProperties.cpp include/ui_mcam.h $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/EditProperties.cpp

EditProperties_moc.o: src/EditProperties_moc.cpp $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/EditProperties_moc.cpp

Main.o: src/Main.cpp $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/Main.cpp

MCam.o: src/MCam.cpp  $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/MCam.cpp

ConfigReader.o: src/MCam.cpp  $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/ConfigReader.cpp

MCamCameraIF.o: src/MCamCameraIF.cpp  $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/MCamCameraIF.cpp

MCamUtil.o: src/MCamUtil.cpp  $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/MCamUtil.cpp

MCamImage.o: src/MCamImage.cpp  $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/MCamImage.cpp

MCamTest.o: src/MCamTest.cpp  $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/MCamTest.cpp

mcamlogger.o: src/mcamlogger.c 
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/mcamlogger.c

clean:
	    cd ../../fw/cam_usb/axcam && $(MAKE) clean
	    - rm mcam *.o  src/Application_moc.cpp src/MCamImage_moc.cpp src/MCamCameraIF_moc.cpp src/EditProperties_moc.cpp include/ui_mcam.h *.log

