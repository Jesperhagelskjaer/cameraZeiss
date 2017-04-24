/*
 * Main.cpp
 *
 *  Created on: 11.11.2013
 *      Author: horst
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */

#include <QtGui/QApplication>
#include <Application.hpp>

int main(int argc, char *argv[])
{

    QApplication application(argc, argv);
    Application window;
    window.show();
    return application.exec();

}
