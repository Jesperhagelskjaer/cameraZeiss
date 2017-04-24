/*
 * EditProperties.hpp
 *
 *  Created on: 11.11.2013
 *      Author: ggraf
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */

#ifndef EDITPROPERTIES_HPP_
#define EDITPROPERTIES_HPP_

#include "mcam.h"

#include <QDialog>
#include <QPushButton>
#include <QHBoxLayout>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QRadioButton>
#include <QGroupBox>
#include <QSpinBox>
#include <QLabel>

class EditProperties: public QDialog
{
    Q_OBJECT

public:
    QDialog *properties;
    QPushButton *cancelButton;
    QPushButton *set2defaultButton;
    QPushButton *saveButton;
    QHBoxLayout *buttonsLayout;
    QVBoxLayout *mainLayout;

    QGroupBox *loglevelGroupBox;

    QRadioButton *logRadioNo;
    QRadioButton *logRadioRel;
    QRadioButton *logRadioDev;
    QRadioButton *logRadioAll;

    QVBoxLayout *logvbox;

    QVBoxLayout *updateLayout;

    QCheckBox *resetCameras;
    QComboBox *logLevelCombo;
    QCheckBox *appendLogfile;

    QGroupBox *progSpeedGroupBox;

    QRadioButton *progRadioSafe;
    QRadioButton *progRadioNormal;
    QRadioButton *progRadioFast;
    QLabel *workersLabel;
    QSpinBox *workers;
    QLabel *hint;
    QVBoxLayout *progvbox;

    void exec();
    EditProperties(Ui::Application *app);
    virtual ~EditProperties();

private slots:
    void handleCancelButton();
    void handleSetToDefaultButton();
    void handleSaveButton();
};

#endif /* EDITPROPERTIES_HPP_ */
