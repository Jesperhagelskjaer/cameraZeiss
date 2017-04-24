all: ccam

HFILES=include/ccam.h

CXX_FLAGS = -lpthread -std=c++0x $(CXX_DEBUG_FLAGS) -DNDEBUG  -I/usr/include/libusb-1.0 $(QTINCLUDE) -Iinclude -I../api/include
CXX_DEFINES = -DQT_CORE_LIB -DQT_GUI_LIB -DQT_NO_DEBUG

ccam: ccam.o $(HFILES)
	   c++  -std=c++0x $(LINK_DEBUG_FLAGS) -DNDEBUG  ccam.o  -o ccam -rdynamic -lrt -ldl -ltiff -lusb-1.0 ../api/lib/libaxcam64.so -Wl,-rpath,../api/lib


ccam.o: src/ccam.c  $(HFILES)
	   c++ $(CXX_DEFINES) $(CXX_FLAGS) -c src/ccam.c

clean:
	    - rm ccam *.o 

