/*
 * Application.hpp
 *
 *  Created on:  Oct 21, 2011
 *      Author: ggraf
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */

#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#include "mcam.h"

#include "MCam.hpp"
#include "EditProperties.hpp"
#include "MCamCameraIF.hpp"
#include "MCamImage.hpp"
#include "MCamUtil.hpp"
#include "MCamTest.hpp"
#include "MCamRemote.hpp"

#define SLIDER_LOG_BASE 			1.00011513703

#define WHITE_POINT_DEFAULT_RED     192
#define WHITE_POINT_DEFAULT_GREEN   255
#define WHITE_POINT_DEFAULT_BLUE    129

class Application: public QMainWindow
{
    Q_OBJECT

    bool loadSettingsPending;
    long cameraIndex;
    char logPath[MAX_PATH * 2];
    char propertyPath[MAX_PATH * 2];
    EditProperties *propertyDialog;

    void closeEvent(QCloseEvent *bar);

public:

    QMessageBox msgBox;
    MCamCameraIF *thisCameraIFPtr;
    MCamImage *thisMCamImagePtr;
	MCamRemote *thisMCamRemotePtr;
    MCamTest *thisMCamTestPtr;
    int cameraComboIndexMapping[MCAM_MAX_NO_OF_CAMERAS];

    Application(QWidget *parent = 0);
    ~Application();
    void calcLogAndPropertyPath();
    MCamCameraIF* getCameraIF();
    long getCameraIndex();
    void executePaintImage(QImage *image);
    void doLoadSettings();
    long singleShot();
    long continuousShotStartStop(bool start);
    Ui::Application *getUi();
    MCamImage *getMcamImagePtr();
    long cameraIndexActiveContShotRestart;
    bool isTriggerEnabled();
    void updateGPOGui();
    void setGPOParameter();
    bool contShotRestartPending;
    int currentExposureUnit;
	RECT getCurrentFrameSize();

signals:
    void paintImage(const QImage*);

private slots:
    void open();
    void save();
    void exit();
    void updateDevicesMenu();
    void updateCameraIndex();
    void updateDevicesComboBox();
    void doSingleShot();
    void doContinuousShot(bool start);
    void loadSettings();
    void startStressTest();
    void setImage(const QImage*);
    void doLowQualityDemosaicing(bool enabled);
    void doMediumQualityDemosaicing(bool enabled);
    void doHighQualityDemosaicing(bool enabled);
    void doTileAdjustmentOff(bool enabled);
    void doTileAdjustmentLinear(bool enabled);
    void doTileAdjustmentBilinear(bool enabled);
    void doLineFlickerSuppressionOff(bool enabled);
    void doLineFlickerSuppressionLinear(bool enabled);
    void doLineFlickerSuppressionBilinear(bool enabled);
    void calcBlackReference();
    void doBlackReference(bool enabled);
    void saveBlackReference();
    void restoreBlackReference();
    void calcWhiteReference();
    void doWhiteReference(bool enabled);
    void saveWhiteReference();
    void restoreWhiteReference();
    void doLinGainImage(bool enabled);
    void zoomIn();
    void zoomOut();
    void normalSize();
    void fitToWindow();
    void histogram();
    void editProperties();
    void about();
    void colorTemperatureChanged(int value);
    void sliderExposureChanged(int value);
    void spinBoxValueChanged(int value);
    void redColorChanged(int value);
    void greenColorChanged(int value);
    void blueColorChanged(int value);
    void handleResetColorButton();
    void handleSaveColorButton();
    void handleDefaultColorButton();
    void pixelClockSelected(int);
    void cameraSelectedComboBox(int index);
    void binningSelected(int);
    void portsSelected(int);
    void handleFullROIButton();
    void handleApplyROIButton();
    void handleContShotButton();
    void negativePolarityChecked(bool enabled);
    void levelTriggerAndDebounceChecked(bool enabled);
    // gpo
    void spinBoxTriggerDelayValueChanged(int value);
    void gpoIndexSelectedComboBox(int value);
    void gpoSrcSelectedComboBox(int value);
    void gpoInvertedChecked(bool enabled);
    void gpoPulseSelectedComboBox(int value);
    void gpoDepaySelectedComboBox(int value);

    void noDiscardModeChecked(bool enabled);
    void HDRModeChecked(bool enabled);
    void HighRateModeChecked(bool enabled);
    void exposureTimeTimeUnitSelectedComboBox(int value);
    void handleCompressionModeChange(int value);
    void updateCompressionMode();
    void updateHighRateModeChecked();
    void BufferChecked(bool enabled);
    void updateBufferChecked();

public:
    bool isBufferEnabledCurrentCamera();

public slots:
    void setBusyLock(bool enabled);
    void doUpdateDevices();
    int showMessageBox(QString message);
    int dismissMessageBox();
    int selectCamera(long cameraIndex);
    void updateTransferRate(QString rateStr);
    void contShotStart(bool start);
    void handleTriggerMode(int index);
    void updateCameraGUIParamter(long cameraIndex);
    void cameraSelected(long cameraIndex);
	void updateCost(long cost);

private:
    void updateActions();
    void scaleImage(double factor);
    void adjustScrollBar(QScrollBar* scrollBar, double factor);
    void updateTileAdjustmentSetting();
    void updateLineFlickerSuppresionSetting();
    bool cameraBufferEnabled;

    Ui::Application *ui;
    double scaleFactor;
    long currentExposureValue;

    MCam mCam;


};

#endif // MCAM_HPP
