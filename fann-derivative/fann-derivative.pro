TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

LIBS += -ldoublefann

QMAKE_CXXFLAGS -= -O1
QMAKE_CXXFLAGS -= -O2
