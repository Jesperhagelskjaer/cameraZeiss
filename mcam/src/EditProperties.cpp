/*
 * EditProperties.cpp
 *
 *  Created on: 11.11.2013
 *      Author: ggraf
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 *
 */

#include "EditProperties.hpp"
#include <QRadioButton>
#include <QMessageBox>
#include <QCheckBox>
#include <QLabel>
#include <QSpinBox>

#include <ConfigReader.hpp>
#define PROPERTY_FILE 				"axcam.properties"

// defaults --> see also handleSetToDefaultButton() below
#define RESET_ONS_STARTUP_DEFAULT 		1
#define PROG_SPPED_DEFAULT 				2
#define LOG_LEVEL_DEFAULT				"1+"
#define COLOR_MATRIX_WORKERS_DEFAULT 	2


EditProperties::EditProperties(Ui::Application *app) {

	ConfigReader config(PROPERTY_FILE);

	properties = new QDialog();

	// reset on startup
	resetCameras = new QCheckBox(tr("Reset cameras on DLL startup"));
	int propReset = config.read("resetOnStartup", RESET_ONS_STARTUP_DEFAULT);
	if (propReset)
		resetCameras->setCheckState(Qt::Checked);
	else
		resetCameras->setCheckState(Qt::Unchecked);

	updateLayout = new QVBoxLayout;
	updateLayout->addWidget(resetCameras);

	// loglevel
	string debugLevel;
	config.readInto(debugLevel, "debugLevel", string(LOG_LEVEL_DEFAULT));

	int level = atoi(debugLevel.c_str());
	int append = 0;
	if (strchr(debugLevel.c_str(),'+') != NULL)
		append = 1;
    loglevelGroupBox = new QGroupBox(tr("Logging Level"));

    logRadioNo = new QRadioButton(tr("&No Logging"));
    logRadioRel = new QRadioButton(tr("&Release Logging"));
    logRadioDev = new QRadioButton(tr("&Developer Logging"));
    logRadioAll = new QRadioButton(tr("&All Logging"));

    if (level == 0)
      logRadioNo->setChecked(true);
    else if (level == 1)
      logRadioRel->setChecked(true);
    else if (level == 2)
    	logRadioDev->setChecked(true);
    else if (level >= 3)
    	logRadioAll->setChecked(true);

    logvbox = new QVBoxLayout;
    logvbox->addWidget(logRadioNo);
    logvbox->addWidget(logRadioRel);
    logvbox->addWidget(logRadioDev);
    logvbox->addWidget(logRadioAll);
    loglevelGroupBox->setLayout(logvbox);

    updateLayout->addWidget(loglevelGroupBox);

    appendLogfile = new QCheckBox(tr("Append to logfile (axcam.log) instead of overwriting"));
    if (append)
    	appendLogfile->setCheckState(Qt::Checked);
    else
    	appendLogfile->setCheckState(Qt::Unchecked);

    updateLayout->addWidget(appendLogfile);

    // programming speed
    int speed = config.read("programmingSpeed", PROG_SPPED_DEFAULT);
    progSpeedGroupBox = new QGroupBox(tr("Camera Programming Speed"));

	progRadioSafe = new QRadioButton(tr("&Safe"));
	progRadioNormal = new QRadioButton(tr("&Normal"));
	progRadioFast = new QRadioButton(tr("&Fast"));

	if (speed == 0)
		progRadioSafe->setChecked(true);
	else if (speed == 1)
		progRadioNormal->setChecked(true);
	else
		progRadioFast->setChecked(true);

	progvbox = new QVBoxLayout;
	progvbox->addWidget(progRadioSafe);
	progvbox->addWidget(progRadioNormal);
	progvbox->addWidget(progRadioFast);
	progSpeedGroupBox->setLayout(progvbox);
	updateLayout->addWidget(progSpeedGroupBox);

	int colorMatrixWorkers=COLOR_MATRIX_WORKERS_DEFAULT;
	colorMatrixWorkers = config.read("colorMatrixNumberOfWorkers", COLOR_MATRIX_WORKERS_DEFAULT);

	workersLabel=new QLabel(tr("Number of workers for ColorMatrix calculations"));
	updateLayout->addWidget(workersLabel);
	workers = new QSpinBox();
	workers->setRange(1,4);
	workers->setValue(colorMatrixWorkers);
	updateLayout->addWidget(workers);

	hint=new QLabel(tr("Hint: mcam restart required for changes to take effect!"));
	updateLayout->addWidget(hint);
    // buttons
	cancelButton = new QPushButton(tr("Cancel"), this);
	set2defaultButton = new QPushButton(tr("Reset to Defaults"), this);
	saveButton = new QPushButton(tr("Save"), this);
	buttonsLayout = new QHBoxLayout;
	buttonsLayout->addStretch(1);
	buttonsLayout->addWidget(cancelButton);
	buttonsLayout->addWidget(set2defaultButton);
	buttonsLayout->addWidget(saveButton);

	// Main
	mainLayout = new QVBoxLayout;
	mainLayout->addLayout(updateLayout);
	mainLayout->addLayout(buttonsLayout);

	properties->setLayout(mainLayout);

	properties->setWindowTitle(tr("Axiocam USB 3.0 DLL Properties"));
	connect(cancelButton, SIGNAL(released()), this, SLOT(handleCancelButton()));
	connect(set2defaultButton, SIGNAL(released()), this, SLOT(handleSetToDefaultButton()));
	connect(saveButton, SIGNAL(released()), this, SLOT(handleSaveButton()));
}
void EditProperties::exec() {
	properties->exec();
}

void EditProperties::handleCancelButton() {
	properties->done(0);
}

void EditProperties::handleSetToDefaultButton() {
	resetCameras->setCheckState(Qt::Checked);
	logRadioRel->setChecked(true);
	appendLogfile->setCheckState(Qt::Checked);
	progRadioFast->setChecked(true);
	workers->setValue(COLOR_MATRIX_WORKERS_DEFAULT);
}

void EditProperties::handleSaveButton() {
	bool resetOnStatup = resetCameras->isChecked();
	int loglevel = 1;
	int append = 1;
	int progSpeed = 2;
	int numberWorkers = 2;

	if (logRadioNo->isChecked())
		loglevel = 0;
	if (logRadioRel->isChecked())
		loglevel = 1;
	if (logRadioDev->isChecked())
		loglevel = 2;
	if (logRadioAll->isChecked())
		loglevel = 3;
	if (appendLogfile->isChecked())
		append = 1;
	else
		append = 0;
	if (progRadioSafe->isChecked())
		progSpeed = 0;
	if (progRadioNormal->isChecked())
		progSpeed = 1;
	if (progRadioFast->isChecked())
		progSpeed = 2;
	numberWorkers= workers->value();
	FILE *fp = fopen(PROPERTY_FILE, "wb");
	if (fp != NULL) {
		fprintf(fp, "# properties for axcam64.dll\r\n\r\n");
		fprintf(fp, "# reset all Axiocam USB 3.0 cameras at DLL startup\r\n");
		fprintf(fp, "resetOnStartup = %d\r\n\r\n", resetOnStatup);
		fprintf(fp, "# programming speed 0:safe 1:standard 2:fast\r\n");
		fprintf(fp, "programmingSpeed = %d\r\n\r\n", progSpeed);
		fprintf(fp, "#debug level 0:off 1:release 2:development '+' append file\r\n");
		if (append)
			fprintf(fp, "debugLevel = %d+\r\n\r\n", loglevel);
		else
			fprintf(fp, "debugLevel = %d\r\n\r\n", loglevel);
		fprintf(fp, "# Number of workers for ColorMatrix calculations\r\n");
		fprintf(fp, "colorMatrixNumberOfWorkers = %d\r\n", numberWorkers);
		fclose(fp);
		printf("properties saved to '%s'\n", PROPERTY_FILE);
		properties->done(0);
	} else {
		printf("cannot open '%s', error=%d\r\n", PROPERTY_FILE, errno);
	  QMessageBox msgBox;
	  msgBox.setIcon(QMessageBox::Critical);
	  if (errno == EACCES)
	    msgBox.setText("Error saving properties - Do you have proper access rights?");

	    msgBox.setText("Error saving properties!");
	  msgBox.exec();
	}
}


EditProperties::~EditProperties() {
	// TODO Auto-generated destructor stub
}

