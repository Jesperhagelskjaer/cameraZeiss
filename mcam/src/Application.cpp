/**
 * @file Application.cpp
 * @author ggraf
 * @date 15.02.2015
 *
 * @brief Main module of mcam
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */

#ifdef _WIN32
#include <Shlwapi.h>
#include <shlobj.h>
#endif

#include <QFileDialog>
#include <QMessageBox>
#include <QScrollBar>

#include "Application.hpp"

#define ABOUT_VERSION "About MCam Version 1.62"

Application::Application(QWidget *parent) :
                QMainWindow(parent), ui(new Ui::Application)
{
    MCAM_LOG_INIT("Application::Application")
    char logFilePath[MAX_PATH * 2];
    currentExposureUnit=1000;
    calcLogAndPropertyPath(); // paths end with "\"
    strcpy(logFilePath, logPath);
    strcat(logFilePath, "mcam.log");
    if (strlen(propertyPath) > 0)
        printf("hvorforvirkerdennnumcam property path='%s'\n", propertyPath);
    printf("mcam Logfile in '%s'\n", logFilePath);

    // start logging to mcam.log
    mcamLoggingInit(1, logFilePath, 1000000);
    mcamSetLogLevel(MCAM_DEBUG_ERROR | MCAM_DEBUG_WARN | MCAM_DEBUG_INFO | MCAM_DEBUG_STATUS);

    MCAM_LOGF_STATUS("##################### mcam started #######################");

    cameraIndex = -1;
    scaleFactor = 0;
    currentExposureValue = 0;
    loadSettingsPending = false;
    propertyDialog = NULL;
    contShotRestartPending = false;

    cameraIndexActiveContShotRestart = -1;
    cameraBufferEnabled = false;

    thisCameraIFPtr = new MCamCameraIF(this, propertyPath, logPath); // scan for cameras in here
    thisMCamImagePtr = new MCamImage(this);  // image processing and display
    thisMCamTestPtr = new MCamTest(this);   // simple "stress" test
	thisMCamRemotePtr = new MCamRemote(this); // remote controlled camera thread

    thisCameraIFPtr->loadMcamFileProperties();

    ui->setupUi(this);

    connect(this, SIGNAL(paintImage(const QImage*)), this, SLOT(setImage(const QImage*)));

    connect(ui->colorTemperaturSlider, SIGNAL(valueChanged(int)), this, SLOT(colorTemperatureChanged(int)));

    connect(ui->exposureSlider, SIGNAL(valueChanged(int)), this, SLOT(sliderExposureChanged(int)));
    connect(ui->exposureSpinBox, SIGNAL(valueChanged(int)), this, SLOT(spinBoxValueChanged(int)));

    connect(ui->redSlider, SIGNAL(valueChanged(int)), this, SLOT(redColorChanged(int)));
    connect(ui->greenSlider, SIGNAL(valueChanged(int)), this, SLOT(greenColorChanged(int)));
    connect(ui->blueSlider, SIGNAL(valueChanged(int)), this, SLOT(blueColorChanged(int)));
    connect(ui->resetColorButton, SIGNAL(released()), this, SLOT(handleResetColorButton()));
    connect(ui->saveColorButton, SIGNAL(released()), this, SLOT(handleSaveColorButton()));
    connect(ui->defaultColorButton, SIGNAL(released()), this, SLOT(handleDefaultColorButton()));

    connect(ui->frequencyComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(pixelClockSelected(int)));
    connect(ui->binningComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(binningSelected(int)));
    connect(ui->portComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(portsSelected(int)));

    connect(ui->fullROIButton, SIGNAL(released()), this, SLOT(handleFullROIButton()));
    connect(ui->applyROIButton, SIGNAL(released()), this, SLOT(handleApplyROIButton()));

    connect(ui->singleShotButton, SIGNAL(released()), this, SLOT(doSingleShot()));
    connect(ui->contShotButton, SIGNAL(released()), this, SLOT(handleContShotButton()));
    connect(ui->triggerModeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(handleTriggerMode(int)));

    connect(ui->cameraComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(cameraSelectedComboBox(int)));
    connect(ui->negativePolarityCheckBox, SIGNAL(toggled(bool)), this, SLOT(negativePolarityChecked(bool)));
    connect(ui->levelTriggerCheckBox, SIGNAL(toggled(bool)), this, SLOT(levelTriggerAndDebounceChecked(bool)));
    connect(ui->debounceCheckBox, SIGNAL(toggled(bool)), this, SLOT(levelTriggerAndDebounceChecked(bool)));
    connect(ui->triggerDelaySpinBox, SIGNAL(valueChanged(int)), this, SLOT(spinBoxTriggerDelayValueChanged(int)));

    connect(ui->gpoIndexComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(gpoIndexSelectedComboBox(int)));
    connect(ui->gpoSrcComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(gpoSrcSelectedComboBox(int)));
    connect(ui->gpoInvertedCheckBox, SIGNAL(toggled(bool)), this, SLOT(gpoInvertedChecked(bool)));
    connect(ui->gpoPulseSpinBox, SIGNAL(valueChanged(int)), this, SLOT(gpoPulseSelectedComboBox(int)));
    connect(ui->gpoDelaySpinBox, SIGNAL(valueChanged(int)), this, SLOT(gpoDepaySelectedComboBox(int)));

    //connect(ui->noDiscardModeCheckBox, SIGNAL(toggled(bool)), this, SLOT(noDiscardModeChecked(bool)));
    connect(ui->HDRModeCheckBox, SIGNAL(toggled(bool)), this, SLOT(HDRModeChecked(bool)));
    ui->exposureTimeUnitComboBox->setCurrentIndex(1); //ms
    connect(ui->exposureTimeUnitComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(exposureTimeTimeUnitSelectedComboBox(int)));
    connect(ui->compressionComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(handleCompressionModeChange(int)));

    connect(ui->HighRateCheckBox, SIGNAL(toggled(bool)), this, SLOT(HighRateModeChecked(bool)));
    connect(ui->BufferCheckBox, SIGNAL(toggled(bool)), this, SLOT(BufferChecked(bool)));
    setBusyLock(true);

    thisMCamImagePtr->init();
}

Application::~Application()
{
    MCAM_LOG_INIT("Application::~Application")
    delete ui;
    MCAM_LOGF_STATUS("##################### mcam ends #########################");
    mcamLoggingDeInit();
}

MCamCameraIF* Application::getCameraIF()
{
    return thisCameraIFPtr;
}

Ui::Application *Application::getUi()
{
    return ui;
}

// determine Logfile and Property location depending on Zeiss product used
// for Developer SDK and Linux usage both paths are empty
void Application::calcLogAndPropertyPath()
{
    logPath[0] = '\0';
    propertyPath[0] = '\0';
#ifdef _WIN32
    HMODULE hModule = GetModuleHandle(NULL);
    if (hModule != NULL)
    {
        char ownPth[MAX_PATH];
        char tmpPth[MAX_PATH];
        int i;
        int err= 0;

        ownPth[0]='\0';
        // When passing NULL to GetModuleHandle, it returns handle of exe itself
        GetModuleFileName(hModule,ownPth, (sizeof(ownPth)));
        for (i = 0; i < strlen(ownPth); i++) {
            ownPth[i] = tolower(ownPth[i]);
        }
        // printf("ownPth path='%s'\n", ownPth);

        long result = NOERR;
        TCHAR szPath[MAX_PATH];
        char *mcamPtr = strstr(ownPth, "\\mcam.exe");
        if (mcamPtr != NULL)
        *mcamPtr='\0';
        strcpy(tmpPth, ownPth);
        strcat(tmpPth, "\\0000");

        if (_access(tmpPth, 06) == 0) {
            strcpy(propertyPath, ownPth);
            if ( SUCCEEDED( SHGetFolderPath( NULL, CSIDL_APPDATA, NULL, 0, szPath ) ) )
            {
                // Append product-specific path
                PathAppend( szPath, "\\Carl Zeiss\\AxioVs40\\Profiles\\Default\\" );
                // printf("SHGetFolderPath = %s", szPath);
            }
            strcpy(logPath, szPath);
        }
        strcpy(tmpPth, ownPth);
        strcat(tmpPth, "\\ZEN.exe");
        if (_access(tmpPth, 06) == 0) {
            strcpy(propertyPath, ownPth);
            
            char *homeStr= (char *) getenv("APPDATA");
            char *programDataStr= (char *) getenv("ProgramData");
            
            if (programDataStr != NULL) {
                strcpy(logPath, programDataStr);
                strcat(logPath, "\\Carl Zeiss\\Logging\\");
                if (_access(logPath, 06) != 0) {
                    // cannot access ZEN 2.3 logpath -> use former log path
                    if (homeStr != NULL) {
                        strcpy(logPath, homeStr);
                        strcat(logPath, "\\Carl Zeiss\\ZEN\\");
                    }
                }  
            }
        }
    }
    if (strlen(logPath) > 0) {
        if (_access(logPath, 06) != 0) { // read / write
            logPath[0]='\0';
            if (_access(".", 06) != 0) {
                char *homeStr= (char *) getenv("APPDATA");
                if (homeStr != NULL)
                strcpy(logPath, homeStr);
                strcat(logPath, "\\");
            }
            printf("logPath no write access default to -> logPath='%s'\n", logPath);
        }
    }
    if (strlen((const char*)logPath) == 0)
      strcpy(logPath,".\\");
    if (strlen((const char*)propertyPath) == 0)
      strcpy(propertyPath, ".\\");
#else
    if (strlen((const char*) logPath) == 0)
        strcpy(logPath, "./");
    if (strlen((const char*) propertyPath) == 0)
        strcpy(propertyPath, "./");
#endif
}

int Application::showMessageBox(QString message)
{
    msgBox.setText(message);
    msgBox.setWindowModality(Qt::WindowModal);
    msgBox.show();
    return 0;
}

int Application::dismissMessageBox()
{
    msgBox.close();
    return 0;
}

// GUI callback
void Application::loadSettings()
{
    MCAM_LOG_INIT("Application::loadSettings")
    doLoadSettings();
    MCAM_LOGF_STATUS("settings loaded");
}

void Application::doLoadSettings()
{
    MCAM_LOG_INIT("Application::doLoadSettings")
    MCAM_LOGF_STATUS("load for cameraIndex=%ld", cameraIndex);
    loadSettingsPending = true;
    handleResetColorButton();
    thisCameraIFPtr->loadMcamFileProperties();
    if (cameraIndex >= 0) {
        thisCameraIFPtr->setCameraDefaults(cameraIndex);
        updateCameraGUIParamter(cameraIndex); // values from camera
    }
    // property unit in us
    currentExposureValue = thisCameraIFPtr->getMcamPropertyPtr()->exposureTime / currentExposureUnit;
    ui->exposureSpinBox->setValue(thisCameraIFPtr->getMcamPropertyPtr()->exposureTime / currentExposureUnit);
    loadSettingsPending = false;
}

int Application::selectCamera(long cameraIndex)
{
    MCAM_LOG_INIT("Application::selectCamera")
    long result = NOERR;
    long index = 0;
    long binning = 1;
    bool right = false;
    bool bottom = false;
    RECT rcSize;

    // update UI
    this->cameraIndex = cameraIndex;
    MCAM_LOGF_INFO("selected cameraIndex=%ld", cameraIndex);

    result = McammGetCurrentPixelClock(cameraIndex, &index);
    if (result == NOERR)
        ui->frequencyComboBox->setCurrentIndex(index);

    result = McammGetCurrentFrameSize(cameraIndex, &rcSize);
    if (result == NOERR) {
        ui->posX->setText(QString::number(rcSize.top));
        ui->posY->setText(QString::number(rcSize.left));
        ui->sizeX->setText(QString::number(rcSize.right - rcSize.left));
        ui->sizeY->setText(QString::number(rcSize.bottom - rcSize.top));
    }

    result = McammGetCurrentBinning(cameraIndex, &binning);
    if (result == NOERR) {
        ui->binningComboBox->setCurrentIndex(binning - 1);
    }

    result = McammCurrentUsedSensorTaps(cameraIndex, &right, &bottom);
    if (result == NOERR) {
        int index = 0;
        if (right)
            index = 1;
        if (right && bottom)
            index = 2;
        ui->portComboBox->setCurrentIndex(index);
    }
    handleResetColorButton();

    // set default sensor temperature (cooling)
    McammResetSensorTemperature(cameraIndex);

    return this->cameraIndex;
}


long Application::getCameraIndex()
{
    return cameraIndex;
}

// paint image in context of GUI thread
void Application::executePaintImage(QImage *image)
{
    emit paintImage(image);
}

// display image in context of GUI thread
void Application::setImage(const QImage* image)
{
    thisMCamImagePtr->setImage(image);
}
MCamImage *Application::getMcamImagePtr()
{
    return thisMCamImagePtr;
}

// load and display an image from file system
void Application::open()
{
    MCAM_LOG_INIT("Application::open")
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::currentPath());

    if (!fileName.isEmpty()) {
        QImage* image = new QImage(fileName);
        if (image->isNull()) {
            QMessageBox::information(this, tr("Image Viewer"), tr("Cannot load %1.").arg(fileName));
            return;
        }
        if (thisMCamImagePtr->setImage(image)) {
            scaleFactor = 1.0;
            updateActions();
        }
    } else
        MCAM_LOGF_ERROR("no file name specified");
}

// store image to file system
void Application::save()
{
    QString filter = "*.png;; *.xpm;; *.jpg;; *.jpeg;; *.bmp;; *.tif";
    QString fileName = QFileDialog::getSaveFileName(NULL, tr("Save file"), "MCamImage.tif", filter);
    if (!fileName.isNull() && !fileName.isEmpty()) {
        QString suffix = QFileInfo(fileName).suffix();
        if (filter.contains(suffix)) {
            if (ui->imageLabel->pixmap() != NULL) {
                ui->imageLabel->pixmap()->save(fileName);
            }
        } else {
            QMessageBox::warning(this, tr("Unsupported Format"), tr("<p>Please select a supported format from file selectors list.</p>"));
        }
    }
}

void Application::exit()
{
    MCAM_LOG_INIT("Application::exit")
    MCAM_LOGF_STATUS("mcam exit called");
    if (thisMCamImagePtr->isContShotRunning()) {
        continuousShotStartStop(false);
    }
    thisMCamImagePtr->deInit();
    thisCameraIFPtr->deInit();
    QApplication::exit();
}

void Application::closeEvent(QCloseEvent *bar)
{
    exit();
}

void Application::doLinGainImage(bool enabled)
{
    thisMCamImagePtr->setHighGain(enabled);
}

void Application::zoomIn()
{
    scaleImage(1.25);
}

void Application::zoomOut()
{
    scaleImage(0.8);
}

void Application::normalSize()
{
    ui->imageLabel->adjustSize();
    scaleFactor = 1.0;
}

void Application::fitToWindow()
{
    bool fitToWindow = ui->actionFitToWindow->isChecked();
    ui->scrollArea->setWidgetResizable(fitToWindow);
    if (!fitToWindow) {
        normalSize();
    }
    updateActions();
}

void Application::editProperties()
{
    if (propertyDialog == NULL)
        propertyDialog = new EditProperties((Ui::Application*) this);
    propertyDialog->exec();
}

void Application::about()
{
    QMessageBox::about(this, tr(ABOUT_VERSION),
                    tr(
                                    "<p>MCam is a diagnostic program for Zeiss Axiocam USB 3.0 cameras</p>\n\
<p>MCam uses the 'Qt Library' according to the LGPL 2.1<br>(GNU Lesser General Public License Version 2.1)<br>\
See also the included license text file: 'lgpl-2.1.txt'</p>"));
}

void Application::updateActions()
{
    ui->actionZoomIn->setEnabled(!ui->actionFitToWindow->isChecked());
    ui->actionZoomOut->setEnabled(!ui->actionFitToWindow->isChecked());
    ui->actionNormalSize->setEnabled(!ui->actionFitToWindow->isChecked());
    ui->actionSave->setEnabled(ui->imageLabel->pixmap() != NULL);
}


// public
void Application::doUpdateDevices()
{
    MCAM_LOG_INIT("Application::doUpdateDevices")
    long activeCameraIndex = thisMCamImagePtr->getActiveCameraIndex();

    updateDevicesMenu();
    updateDevicesComboBox();

    // try to stop / restart ContShot
    bool running = thisMCamImagePtr->isContShotRunning();
    bool initialized = false;

    if (activeCameraIndex >= 0) {
        initialized = thisCameraIFPtr->isCameraInitialized(activeCameraIndex);
		MCAM_LOGF_INFO("running=%d activeCameraIndex=%d isCameraInitialized=%d", running, activeCameraIndex, initialized);
		if ((!initialized) || (activeCameraIndex != cameraIndex)) {
			if (running) {
				// camera not active or other camera was selected -> stop cont shot
				MCAM_LOGF_INFO("stop cont shot for cameraIndex=%d", activeCameraIndex);
				doContinuousShot(false);
				cameraIndexActiveContShotRestart = activeCameraIndex;
			}
		} else {
			  // not running
			  if (initialized) {
				  // camera active -> need restart?
				  if (cameraIndexActiveContShotRestart >= 0) {
					  MCAM_LOGF_INFO("restart cont shot for cameraIndexActiveContShotRestart=%d", cameraIndexActiveContShotRestart);
					  if (ui->triggerModeComboBox->currentIndex() == 1)
						  thisMCamImagePtr->setSoftwareTrigger(cameraIndex, true);
					  doContinuousShot(true);
					  cameraIndexActiveContShotRestart = -1;
				  }
			  }
		}
    }
    if (thisCameraIFPtr->getNumberOfCameras() > 0)
      setBusyLock(false);
    else
      setBusyLock(true);
}

// maintain device menu
void Application::updateDevicesMenu()
{
    long result = NOERR;
    static bool firstStart = true;
    QAction* actionDevice;

    // remove all camera entries
    QList<QAction*> actionsDevice = ui->menuDevice->actions();
    for (int i = 0; i < actionsDevice.size(); ++i) {
        actionDevice = actionsDevice.at(i);
        if (!actionDevice->isSeparator()) {
            ui->menuDevice->removeAction(actionDevice);
            ui->actionGroupDevices->removeAction(actionDevice);
        }
    }
    long availableCameras = 0;
    // add new entries
    for (int i = 0; i < MCAM_MAX_NO_OF_CAMERAS; ++i) {
        if (thisCameraIFPtr->isCameraInitialized(i)) {
            char buffer[1024];
            availableCameras++;
            thisCameraIFPtr->getDeviceString(i, buffer);
            actionDevice = ui->menuDevice->addAction(QString(buffer));
            actionDevice->setCheckable(true);
            actionDevice->setProperty("cameraIndex", i);
            ui->actionGroupDevices->addAction(actionDevice);
            connect(actionDevice, SIGNAL(changed()), this, SLOT(updateCameraIndex()));

            if (firstStart || cameraIndex < 0) {
                // use 1st camera in list
                firstStart = false;
                actionDevice->setChecked(true);
                cameraIndex = i;
            }
            if (i == cameraIndex && result == NOERR)
                actionDevice->setChecked(true);
            updateCameraGUIParamter(cameraIndex);
        }
    }

    if (availableCameras == 0) {
        if (!ui->menuDevice->actions().contains(ui->actionNoDevice)) {
            ui->actionGroupDevices->addAction(ui->actionNoDevice);
            cameraIndex = -1;
        }
    }
    firstStart = false;
}
// update the entries in the devices combo box
void Application::updateDevicesComboBox()
{
  long result = NOERR;
  int comboIndex = 0;
  ui->cameraComboBox->clear();
  for (int i = 0; i < MCAM_MAX_NO_OF_CAMERAS; ++i)
    cameraComboIndexMapping[i] = -1;

  for (int i = 0; i < MCAM_MAX_NO_OF_CAMERAS; ++i) {
    if (thisCameraIFPtr->isCameraInitialized(i)) {
      char buffer[1024];
      thisCameraIFPtr->getDeviceString(i, buffer);
      ui->cameraComboBox->addItem(QString(buffer));
      cameraComboIndexMapping[comboIndex++] = i;
    }
  }
}

// GUI call
// Camera was selected combo box
void Application::cameraSelectedComboBox (int index)
{
  MCAM_LOG_INIT("Application::cameraSelectedComboBox")
  if (index>= 0 && index < MCAM_MAX_NO_OF_CAMERAS) {
    MCAM_LOGF_INFO("cameraSelectedComboBox index=%d -> cameraIndex = %d", index, cameraComboIndexMapping[index]);
    if (cameraComboIndexMapping[index] >= 0) {
      if (cameraComboIndexMapping[index] != cameraIndex) {
           if (cameraIndex >=0)
             cameraIndexActiveContShotRestart = -1; // other camera
           // another camera was selected (no connect / disconnect)
           contShotRestartPending = thisMCamImagePtr->isContShotRunning();
           if (contShotRestartPending)
             doContinuousShot(false);
           cameraIndex = cameraComboIndexMapping[index];
           // inform image class --> callback -->cameraSelected()
           thisMCamImagePtr->selectCamera(cameraIndex);
           updateCameraGUIParamter(cameraIndex);
           updateDevicesMenu();
       }
    }
  }
}

// GUI menu selection
// called if a camera was selected in camera menu or camera was added
// switch current camera to camera with selected index
void Application::updateCameraIndex()
{
    MCAM_LOG_INIT("Application::updateCameraIndex")
    long result = NOERR;
    QAction *qa = ui->actionGroupDevices->checkedAction();
    if (qa == NULL)
        return;
    int actionDeviceIndex = qa->property("cameraIndex").toInt();
    MCAM_LOGF_INFO("updateCameraIndex called actionDeviceIndex=%ld current=%ld", actionDeviceIndex, cameraIndex);

    if (actionDeviceIndex != cameraIndex && actionDeviceIndex >= 0) {
        if (cameraIndex >= 0)
          cameraIndexActiveContShotRestart = -1; // other camera

        // another camera was selected (no connect / disconnect)
        contShotRestartPending = thisMCamImagePtr->isContShotRunning();
        if (contShotRestartPending)
          doContinuousShot(false);
        cameraIndex = actionDeviceIndex;
        MCAM_LOGF_INFO("selectCamera cameraIndex=%ld", cameraIndex);

        // inform image class --> callback -->cameraSelected()
        thisMCamImagePtr->selectCamera(cameraIndex);
        updateCameraGUIParamter(cameraIndex);
        updateTileAdjustmentSetting();
        updateLineFlickerSuppresionSetting();
        updateHighRateModeChecked();
        updateBufferChecked();

        // update combo box
        int comboIndex = -1;
        for (int i = 0; i < MCAM_MAX_NO_OF_CAMERAS; ++i) {
          if (cameraComboIndexMapping[i] == cameraIndex) {
            comboIndex = i;
            break;
          }
        }
        if (comboIndex >= 0)
          ui->cameraComboBox->setCurrentIndex(comboIndex);
    }
}

// called via signal from McamImage if selection of new camera was completed
void Application::cameraSelected(long cameraIndex)
{
  MCAM_LOG_INIT("Application::cameraSelected")

  MCAM_LOGF_INFO("Camera cameraIndex=%ld", cameraIndex);
  if (contShotRestartPending) {
      contShotRestartPending = false;
      doContinuousShot(true);
  }
}

void Application::startStressTest()
{
    cameraIndexActiveContShotRestart = -1;
    thisMCamTestPtr->startStressTest();
}

long Application::singleShot()
{
    cameraIndexActiveContShotRestart = -1;
    return thisMCamImagePtr->doSingleShot(cameraIndex);
}

// GUI menu callback
void Application::doSingleShot()
{
    //KBE??? 
	//thisMCamRemotePtr->createTestImage();
	//printf("Application::doSingleShot\r\n");
	//thisMCamRemotePtr->startRemoteThread();
	thisMCamImagePtr->doSingleShot(cameraIndex);
    updateActions();
}

// public
long Application::continuousShotStartStop(bool start)
{
    return thisMCamImagePtr->doContinuousShot(cameraIndex, start);
}

void Application::handleContShotButton()
{
    cameraIndexActiveContShotRestart = -1;
    if (thisMCamImagePtr->isContShotRunning()) {
        doContinuousShot(false);
    } else {
        doContinuousShot(true);
        updateActions();
    }
}

// GUI callback
void Application::doContinuousShot(bool start)
{
    MCAM_LOG_INIT("Application::doContinuousShot")
	long increment = 0;

    MCAM_LOGF_INFO("start=%d running=%d", start, thisMCamImagePtr->isContShotRunning());
    if (start == thisMCamImagePtr->isContShotRunning())
        return;

    McammGetCurrentImageNumberIncrement(cameraIndex, &increment);
    MCAM_LOGF_INFO("image increment = %d images", increment);

    long result = thisMCamImagePtr->doContinuousShot(cameraIndex, start);
    if (result != NOERR)
      MCAM_LOGF_ERROR("cameraIndex=%ld, doContinuousShot failed. Result=%ld", cameraIndex, result);
    if ((result != NOERR) && start) // stop always possible
        return; // start did not work
    if (start) {
        ui->contShotButton->setText(QString("StopContShot"));
    } else {
        ui->contShotButton->setText(QString("ContShot"));
    }

    QAction* actionDevice;
    // find cont shot menu entry
    QList<QAction*> actionsDevice = ui->menuCamera->actions();

    // update menu entry
    for (int i = 0; i < actionsDevice.size(); ++i) {
        actionDevice = actionsDevice.at(i);
        if (actionDevice == ui->actionContinuousShot) {
            actionDevice->setChecked(start);
        }
    }
    MCAM_LOGF_INFO("start=%d, running=%d, done ok",start, thisMCamImagePtr->isContShotRunning());
}

// GUI menu callback
void Application::doLowQualityDemosaicing(bool enabled)
{
    MCAM_LOG_INIT("Application::doLowQualityDemosaicing")
    long result = McammSetResolution(cameraIndex, -1); // -1
    if (result == NOERR) {
        MCAM_LOGF_INFO("Low quality image processing enabled");
    }
}

// GUI menu callback
void Application::doMediumQualityDemosaicing(bool enabled)
{
    MCAM_LOG_INIT("Application::doMediumQualityDemosaicing")
    long result = McammSetResolution(cameraIndex, 0); // 0
    if (result == NOERR) {
        MCAM_LOGF_INFO("Medium quality image processing enabled");
    }
}

// GUI menu callback
void Application::doHighQualityDemosaicing(bool enabled)
{
    MCAM_LOG_INIT("Application::doHighQualityDemosaicing")
    long result = McammSetResolution(cameraIndex, 1); //1
    if (result == NOERR) {
        MCAM_LOGF_INFO("High quality image processing enabled");
    }
}

// GUI menu callback
void Application::doTileAdjustmentOff(bool enabled)
{
    if(enabled) {
        McammSetTileAdjustmentMode(cameraIndex,  mcammTileAdjustmentOff);
    }
}

// GUI menu callback
void Application::doTileAdjustmentLinear(bool enabled)
{
    if(enabled) {
        McammSetTileAdjustmentMode(cameraIndex, mcammTileAdjustmentLinear);
    }
}

// GUI menu callback
void Application::doTileAdjustmentBilinear(bool enabled)
{
    if(enabled) {
        //FIXME: this is because the enum in mcam_zei_ex.h is incomplete
        McammSetTileAdjustmentMode(cameraIndex, mcammTileAdjustmentBiLinear);
    }
}

// GUI menu callback
void Application::doLineFlickerSuppressionOff(bool enabled)
{
    if(enabled) {
        McammSetLineFlickerSuppressionMode(cameraIndex, mcammLineFlickerSuppressionOff);
    }
}

// GUI menu callback
void Application::doLineFlickerSuppressionLinear(bool enabled)
{
    if(enabled) {
        McammSetLineFlickerSuppressionMode(cameraIndex, mcammLineFlickerSuppressionLinear);
    }
}

// GUI menu callback
void Application::doLineFlickerSuppressionBilinear(bool enabled)
{
    if(enabled) {
        McammSetLineFlickerSuppressionMode(cameraIndex, mcammLineFlickerSuppressionBiLinear);
    }
}

// GUI menu callback
void Application::calcBlackReference()
{
    MCAM_LOG_INIT("Application::calcBlackReference")
    if (!thisMCamImagePtr->isContShotRunning()) {
        int result = McammCalculateBlackRefEx(cameraIndex, NULL, NULL);
        if (result != NOERR) {
            char msgBuffer[1024];
            sprintf(msgBuffer, "<p>Calcuation of BlackReference failed: Error: '%s' (%d).</p>", MCamCameraIF::mcamGetErrorString(result),
                            result);
            QMessageBox::warning(this, tr("BlackReference Calculation Error"), tr(msgBuffer));
            MCAM_LOGF_ERROR("%s", msgBuffer);
        } else
            MCAM_LOGF_INFO("McammCalculateBlackRefEx returned result=%d", result);
    } else {
        QMessageBox::warning(this, tr("BlackReference Calculation Error"),
                        tr("Cannot calculate black reference: Continuous Shot is running!"));
        MCAM_LOGF_ERROR("Cannot calculate black reference: Continuous Shot currently running!");
    }
}

// GUI menu callback
void Application::doBlackReference(bool enabled)
{
    MCAM_LOG_INIT("Application::doBlackReference")
    int result = McammSetBlackRef(cameraIndex, enabled);
    MCAM_LOGF_INFO("McammSetBlackRef returned result=%d", result);
}

// GUI menu callback
void Application::saveBlackReference()
{
    MCAM_LOG_INIT("Application::saveBlackReference")
    // creating data buffer
    long width, height;
    eMcamScanMode mode;
    int result = McammGetResolutionValues(cameraIndex, 0, &width, &height, &mode);
    if (result == NOERR) {
        long blackRefByteSize;
        result = McammGetBlackReferenceDataSize(cameraIndex, &blackRefByteSize);
        if (result == NOERR) {
            unsigned short* blackRef = new unsigned short[blackRefByteSize / sizeof(unsigned short)];
            result = McammSaveBlackRef(cameraIndex, blackRef, blackRefByteSize);
            if (result == NOERR) {
                QFile file("BlackReference.dat");
                if (file.open(QIODevice::WriteOnly)) {
                    QDataStream stream(&file);
                    stream.setVersion(QDataStream::Qt_4_6);
                    QByteArray byteArray = QByteArray::fromRawData(reinterpret_cast<const char*>(blackRef), blackRefByteSize);
                    stream << byteArray;
                    file.close();
                    delete blackRef;
                    MCAM_LOGF_INFO("Black reference saved to: '%s", file.fileName().toStdString().c_str());
                } else {
                    MCAM_LOGF_ERROR("Could not open file to write black reference");
                }
            } else {
                MCAM_LOGF_ERROR("Could not save black reference. Error= %ld", result);
            }
        } else {
            MCAM_LOGF_ERROR("Could not get black reference size. Error=%ld", result);
        }

    } else {
        MCAM_LOGF_ERROR("Could not get resolution values to save black reference. Error=%ld", result);
    }
}

// GUI menu callback
void Application::restoreBlackReference()
{
    MCAM_LOG_INIT("Application::restoreBlackReference")
    // creating data buffer
    long width, height;
    eMcamScanMode mode;
    int result = McammGetResolutionValues(cameraIndex, 0, &width, &height, &mode);
    if (result == NOERR) {
        long blackRefByteSize;
        result = McammGetBlackReferenceDataSize(cameraIndex, &blackRefByteSize);
        if (result == NOERR) {
            unsigned short* blackRef = new unsigned short[blackRefByteSize / sizeof(unsigned short)];
            QFile file("BlackReference.dat");
            if (file.open(QIODevice::ReadOnly)) {
                QDataStream stream(&file);
                stream.setVersion(QDataStream::Qt_4_6);
                stream.readRawData(new char[4], 4); // TODO check how to disable stream header
                stream.readRawData(reinterpret_cast<char*>(blackRef), blackRefByteSize);
                file.close();
                result = McammRestoreBlackRef(cameraIndex, blackRef, blackRefByteSize);
                if (result == NOERR) {
                    MCAM_LOGF_ERROR("Black reference restored from: '%s'", file.fileName().toStdString().c_str());
                    delete blackRef;
                }
            } else {
                MCAM_LOGF_ERROR("Could not open file to restore black reference");
                delete blackRef;
            }
        } else {
            MCAM_LOGF_ERROR("Could not get black reference size. Error=%ld", result);
        }

    } else {
        MCAM_LOGF_ERROR("Could not get resolution values to restore black reference. Error=%ld", result);
    }
}

// GUI menu callback
void Application::calcWhiteReference()
{
    MCAM_LOG_INIT("Application::calcWhiteReference")
    if (!thisMCamImagePtr->isContShotRunning()) {
        int result = McammCalculateWhiteRefEx(cameraIndex, NULL, NULL);
        if (result != NOERR) {
            char msgBuffer[1024];
            sprintf(msgBuffer, "<p>Calculation of WhiteReference failed: Error: '%s' (%d).</p>", MCamCameraIF::mcamGetErrorString(result),
                            result);
            QMessageBox::warning(this, tr("WhiteReference Calculation Error"), tr(msgBuffer));
            MCAM_LOGF_ERROR("%s", msgBuffer);
        }
    } else {
        QMessageBox::warning(this, tr("WhiteReference Calculation Error"),
                        tr("Cannot calculate white reference: Continuous Shot is running!"));
        MCAM_LOGF_ERROR("cannot calculate white reference: Continuous Shot is running!");
    }
}

// GUI menu callback
void Application::doWhiteReference(bool enabled)
{
    MCAM_LOG_INIT("Application::doWhiteReference")
    int result = McammSetWhiteRef(cameraIndex, enabled);
    MCAM_LOGF_INFO("McammSetWhiteRef executed: result=%d", result);
}

// GUI menu callback
void Application::saveWhiteReference()
{
    MCAM_LOG_INIT("Application::saveWhiteReference")
    // creating data buffer
    long width, height;
    eMcamScanMode mode;
    int result = McammGetResolutionValues(cameraIndex, 0, &width, &height, &mode);
    if (result == NOERR) {
        long whiteRefByteSize;
        result = McammGetWhiteReferenceDataSize(cameraIndex, &whiteRefByteSize);
        if (result == NOERR) {
            short* whiteRef = new short[whiteRefByteSize / sizeof(short)];
            result = McammGetWhiteTable(cameraIndex, whiteRef);
            if (result == NOERR) {
                QFile file("WhiteReference.dat");
                if (file.open(QIODevice::WriteOnly)) {
                    QDataStream stream(&file);
                    stream.setVersion(QDataStream::Qt_4_6);
                    QByteArray byteArray = QByteArray::fromRawData(reinterpret_cast<const char*>(whiteRef), whiteRefByteSize);
                    stream << byteArray;
                    file.close();
                    delete whiteRef;
                    MCAM_LOGF_INFO("White reference saved to: '%s'", file.fileName().toStdString().c_str());
                } else {
                    MCAM_LOGF_ERROR("Could not open file to write white reference");
                }
            } else {
                MCAM_LOGF_ERROR("Could not save white reference. Error=%ld", result);
            }
        } else {
            MCAM_LOGF_ERROR("Could not get white reference size. Error=%ld", result);
        }

    } else {
        MCAM_LOGF_ERROR("Could not get resolution values to save white reference. Error=%ld", result);
    }
}

// GUI menu callback
void Application::restoreWhiteReference()
{
    MCAM_LOG_INIT("Application::restoreWhiteReference")
    // creating data buffer
    long width, height;
    eMcamScanMode mode;
    int result = McammGetResolutionValues(cameraIndex, 0, &width, &height, &mode);
    if (result == NOERR) {
        long whiteRefByteSize;
        result = McammGetWhiteReferenceDataSize(cameraIndex, &whiteRefByteSize);
        if (result == NOERR) {
            short* whiteRef = new short[whiteRefByteSize / sizeof(short)];
            QFile file("WhiteReference.dat");
            if (file.open(QIODevice::ReadOnly)) {
                QDataStream stream(&file);
                stream.setVersion(QDataStream::Qt_4_6);
                stream.readRawData(new char[4], 4); // TODO check how to disable stream header
                stream.readRawData(reinterpret_cast<char*>(whiteRef), whiteRefByteSize);
                file.close();
                result = McammSetWhiteTable(cameraIndex, whiteRef);
                if (result == NOERR) {
                    MCAM_LOGF_INFO("White reference restored from:'%s'", file.fileName().toStdString().c_str());
                    delete whiteRef;
                }
            } else {
                MCAM_LOGF_ERROR("Could not open file to restore white reference.");
                delete whiteRef;
            }
        } else {
            MCAM_LOGF_ERROR("Could not get white reference size. Error=", result);
        }

    } else {
        MCAM_LOGF_ERROR("Could not get resolution values to restore white reference. Error=%ld", result);
    }
}

// GUI menu callback
void Application::scaleImage(double factor)
{
    Q_ASSERT(ui->imageLabel->pixmap());
    scaleFactor *= factor;
    ui->imageLabel->resize(scaleFactor * ui->imageLabel->pixmap()->size());

    adjustScrollBar(ui->scrollArea->horizontalScrollBar(), factor);
    adjustScrollBar(ui->scrollArea->verticalScrollBar(), factor);

    ui->actionZoomIn->setEnabled(scaleFactor < 3.0);
    ui->actionZoomOut->setEnabled(scaleFactor > 0.333);
}

void Application::adjustScrollBar(QScrollBar *scrollBar, double factor)
{
    scrollBar->setValue(int(factor * scrollBar->value() + ((factor - 1) * scrollBar->pageStep() / 2)));
}

// GUI callback
void Application::exposureTimeTimeUnitSelectedComboBox(int value) {
	//printf("exposureTimeTimeUnitSelectedComboBox = %d currentExposureUnit=%d currentExposureValue=%ld\n",
	//			value, currentExposureUnit, currentExposureValue);
	if (value == 0) {
		if (currentExposureUnit == 1000)
			currentExposureValue *= 1000;
		else if (currentExposureUnit == 1000000)
			currentExposureValue *= 1000000;
		currentExposureUnit = 1;
	}
	else if(value == 1) {
		if (currentExposureUnit == 1)
			currentExposureValue /= 1000;
		if (currentExposureUnit == 1000000)
			currentExposureValue *= 1000;
		currentExposureUnit = 1000;
	}
	else if (value == 2) {
		if (currentExposureUnit == 1)
			currentExposureValue /= 1000000;
		if (currentExposureUnit == 1000)
			currentExposureValue /= 1000;
		currentExposureUnit = 1000000;
	}
	//printf("NEW currentExposureUnit=%d currentExposureValue=%ld\n",
	//				currentExposureUnit, currentExposureValue);
	updateCameraGUIParamter(cameraIndex);
}

// GUI callback
void Application::sliderExposureChanged(int value)
{  // 0 .. 9999
    double dvalue = pow(SLIDER_LOG_BASE, value) + 0.5;
    currentExposureValue = (long) (dvalue);
    if (ui->exposureSlider->isSliderDown())
        ui->exposureSpinBox->setValue(((int) dvalue));
}

// GUI callback
void Application::spinBoxValueChanged(int value)
{ // 1 ... 100000
    long exposureTime = 0;
    long sliderValue = log10((double) value) / log10(SLIDER_LOG_BASE) + 0.5;
    long result = NOERR;
    long minE,maxE,incE;
    currentExposureValue = value;
    ui->exposureSlider->setValue(sliderValue);
    McammGetExposureRange(cameraIndex, &minE, &maxE, &incE);
    if (currentExposureValue * currentExposureUnit < minE)
    	currentExposureValue = minE;
    if (currentExposureValue * currentExposureUnit > maxE)
    	currentExposureValue = maxE / currentExposureUnit;

    McammSetExposure(cameraIndex, currentExposureValue * currentExposureUnit);
    result = McammGetCurrentExposure(cameraIndex, &exposureTime);
	if (result == NOERR) {
		currentExposureValue = exposureTime / currentExposureUnit;
		ui->exposureSpinBox->setValue(currentExposureValue);
	}
	updateCompressionMode();
}

// GUI callback
void Application::redColorChanged(int value)
{
    double red, green, blue;

    thisCameraIFPtr->getWhitePoint(&red, &green, &blue);
    red = value + 1;

    if (!loadSettingsPending) {
        thisCameraIFPtr->setWhitePoint(red, green, blue);
        thisCameraIFPtr->setWhiteBalance(cameraIndex, red, green, blue);
    }
}

// GUI callback
void Application::greenColorChanged(int value)
{
    double red, green, blue;
    thisCameraIFPtr->getWhitePoint(&red, &green, &blue);
    green = value + 1;
    if (!loadSettingsPending) {
        thisCameraIFPtr->setWhitePoint(red, green, blue);
        thisCameraIFPtr->setWhiteBalance(cameraIndex, red, green, blue);
    }
}

// GUI callback
void Application::blueColorChanged(int value)
{
    double red, green, blue;
    thisCameraIFPtr->getWhitePoint(&red, &green, &blue);
    blue = value + 1;

    if (!loadSettingsPending) {
        thisCameraIFPtr->setWhitePoint(red, green, blue);
        thisCameraIFPtr->setWhiteBalance(cameraIndex, red, green, blue);
    }
}

// reset to program start (mcam.properties) default
// GUI callback
void Application::handleResetColorButton()
{
    double red, green, blue;
    if (!loadSettingsPending) {
        thisCameraIFPtr->resetWhitePoint();
    }
    thisCameraIFPtr->getWhitePoint(&red, &green, &blue);

    ui->colorTemperaturSlider->setValue(50);

    ui->redSlider->setValue(red);
    ui->greenSlider->setValue(green);
    ui->blueSlider->setValue(blue);

    if (!loadSettingsPending) {
        thisCameraIFPtr->setWhitePoint(red, green, blue);
        thisCameraIFPtr->setWhiteBalance(cameraIndex, red, green, blue);
    }
}

// reset to camera default
// GUI callback
void Application::handleDefaultColorButton()
{

    double red, green, blue;
    loadSettingsPending = true;

    red = WHITE_POINT_DEFAULT_RED;
    green = WHITE_POINT_DEFAULT_GREEN;
    blue = WHITE_POINT_DEFAULT_BLUE;

    thisCameraIFPtr->setWhitePoint(red, green, blue);
    thisCameraIFPtr->getWhitePoint(&red, &green, &blue);
    ui->redSlider->setValue(red);
    ui->greenSlider->setValue(green);
    ui->blueSlider->setValue(blue);
    thisCameraIFPtr->setWhiteBalance(cameraIndex, red, green, blue);

    loadSettingsPending = false;
}

// only changed transient values in camera
void Application::colorTemperatureChanged(int value)
{
    thisMCamImagePtr->colorTemperatureChanged(cameraIndex, value);
}

// GUI callback
void Application::handleSaveColorButton()
{
    thisCameraIFPtr->saveMcamProperties();
}

void Application::setBusyLock(bool disabled)  // true: locked  false: unlocked
{
    bool running = thisMCamImagePtr->isContShotRunning();
    BOOL hasPixelClocks = FALSE;
    BOOL hasMultiplePortModes = FALSE;
    BOOL hasHDR = FALSE;
    BOOL hasHighImageRateMode = FALSE;

    MCammHasParameter(cameraIndex, mcammParmPixelClocks, &hasPixelClocks);
    MCammHasParameter(cameraIndex, mcammParmMultiplePortModes, &hasMultiplePortModes);
    MCammHasParameter(cameraIndex, mcammParmHDR, &hasHDR);
    MCammHasParameter(cameraIndex, mcammParmHighImageRateMode, &hasHighImageRateMode);
	
	//KBE!!!
	disabled = false;
	if (!disabled) 
		thisMCamRemotePtr->startRemoteThread();

    ui->menuFile->setEnabled(!disabled);
    ui->menuCamera->setEnabled(!disabled);
    ui->menuImageProcessing->setEnabled(!disabled);
    ui->menuView->setEnabled(!disabled);

    ui->colorTemperaturSlider->setEnabled(!disabled);
    ui->redSlider->setEnabled(!disabled);
    ui->greenSlider->setEnabled(!disabled);
    ui->blueSlider->setEnabled(!disabled);
    ui->saveColorButton->setEnabled(!disabled);
    ui->resetColorButton->setEnabled(!disabled);
    ui->defaultColorButton->setEnabled(!disabled);
    ui->singleShotButton->setEnabled((!disabled) && (!running));
    ui->contShotButton->setEnabled(!disabled);
    ui->triggerModeComboBox->setEnabled((!disabled) && (!running));
    ui->posX->setEnabled(!disabled);
    ui->posY->setEnabled(!disabled);
    ui->sizeX->setEnabled(!disabled);
    ui->sizeY->setEnabled(!disabled);
    ui->fullROIButton->setEnabled(!disabled);
    ui->applyROIButton->setEnabled(!disabled);
    ui->frequencyComboBox->setEnabled((!disabled) && (!running) && hasPixelClocks);
    ui->binningComboBox->setEnabled(!disabled);
    ui->portComboBox->setEnabled((!disabled) && hasMultiplePortModes);
    ui->exposureSpinBox->setEnabled(!disabled);
    ui->exposureSlider->setEnabled(!disabled);

    ui->negativePolarityCheckBox->setEnabled((!disabled) && (!running));
    ui->levelTriggerCheckBox->setEnabled((!disabled) && (!running));
    ui->debounceCheckBox->setEnabled((!disabled) && (!running));
    ui->triggerDelaySpinBox->setEnabled((!disabled) && (!running));

    ui->gpoIndexComboBox->setEnabled(!disabled);
    ui->gpoSrcComboBox->setEnabled(!disabled);
    ui->gpoInvertedCheckBox->setEnabled(!disabled);
    ui->gpoPulseSpinBox->setEnabled(!disabled);
    ui->gpoDelaySpinBox->setEnabled(!disabled);

    ui->compressionComboBox->setEnabled(!disabled);

    //ui->noDiscardModeCheckBox->setEnabled(!disabled);
    if(hasHDR)
        ui->HDRModeCheckBox->setEnabled(!disabled && (!running));
    else
        ui->HDRModeCheckBox->setEnabled(false);
    if (hasHighImageRateMode)
        ui->HighRateCheckBox->setEnabled(!disabled && (!running));
    else
        ui->HighRateCheckBox->setEnabled(false);
    ui->BufferCheckBox->setEnabled(!disabled && (!running));
    ui->exposureTimeUnitComboBox->setEnabled(!disabled);

    enum ECameraType cameraType = MCAM_CAMERA_UNKNOWN;
    if (thisCameraIFPtr->getCameraType(cameraIndex, &cameraType) == NOERR) {
    	if (cameraType != MCAM_CAMERA_CMOS || disabled)
    		ui->menuLineFlickerSuppression->setEnabled(false);
    	else
    		ui->menuLineFlickerSuppression->setEnabled(true);
    	if (cameraType != MCAM_CAMERA_CCD || disabled)
    	    ui->menuTileAdjustment->setEnabled(false);
    	else
    		ui->menuTileAdjustment->setEnabled(true);
    } else {
    	ui->menuLineFlickerSuppression->setEnabled(!disabled);
    	ui->menuTileAdjustment->setEnabled(!disabled);
    }
}

void Application::updateTransferRate(QString rateStr)
{
    ui->fpsStatusLabel->setText(rateStr);
}

void Application::contShotStart(bool start)
{
    BOOL hasPixelClocks = FALSE;
    BOOL hasHDR = FALSE;
    BOOL hasHighImageRateMode = FALSE;

    MCammHasParameter(cameraIndex, mcammParmPixelClocks, &hasPixelClocks);
    MCammHasParameter(cameraIndex, mcammParmHDR, &hasHDR);
    MCammHasParameter(cameraIndex, mcammParmHighImageRateMode, &hasHighImageRateMode);


    if(hasPixelClocks)
        ui->frequencyComboBox->setEnabled(!start);

    ui->singleShotButton->setEnabled(!start);
    ui->triggerModeComboBox->setEnabled(!start);
    ui->negativePolarityCheckBox->setEnabled(!start);
    ui->levelTriggerCheckBox->setEnabled(!start);
    ui->debounceCheckBox->setEnabled(!start);
    ui->triggerDelaySpinBox->setEnabled(!start);
    if (hasHDR)
        ui->HDRModeCheckBox->setEnabled(!start);
    if (hasHighImageRateMode)
        ui->HighRateCheckBox->setEnabled(!start);
}

// GUI callback
void Application::pixelClockSelected(int index)
{
    MCAM_LOG_INIT("Application::pixelClockSelected")
    long currentIndex = 0;
    long numberOfPixelClocks = 1;

    MCAM_LOGF_INFO("changed pixel clock to index=%d", index);
    long result = McammGetNumberOfPixelClocks(cameraIndex, &numberOfPixelClocks);

    if (result == NOERR && index >= 0 && numberOfPixelClocks > 1) {
      long result = McammSetPixelClock(cameraIndex, index);
      if (result != NOERR)
          MCAM_LOGF_ERROR("failed to set pixel clock to index=%d", index);
    }
    result = McammGetCurrentPixelClock(cameraIndex, &currentIndex);
    if (result == NOERR)
    	ui->frequencyComboBox->setCurrentIndex(currentIndex);
}


// GUI callback
void Application::binningSelected(int index)
{
    MCAM_LOG_INIT("Application::binningSelected")
    long currentIndex = 0;
    RECT rcSize;

    MCAM_LOGF_INFO("changed binning to index=%d", index);
    long result = McammSetBinning(cameraIndex, index + 1);
    if (result != NOERR) {
        MCAM_LOGF_ERROR("failed to set binning to index=%d", index);
    }
    result = McammGetCurrentBinning(cameraIndex, &currentIndex);
    if (result == NOERR) {
        ui->binningComboBox->setCurrentIndex(currentIndex - 1);
        result = McammGetCurrentFrameSize(cameraIndex, &rcSize);
        if (result == NOERR) {
            ui->posY->setText(QString::number(rcSize.top));
            ui->posX->setText(QString::number(rcSize.left));
            ui->sizeX->setText(QString::number(rcSize.right - rcSize.left));
            ui->sizeY->setText(QString::number(rcSize.bottom - rcSize.top));
        }
    } else
        MCAM_LOGF_ERROR("failed to get current binning result=%ld", result);
    updateCompressionMode();
}

// GUI callback
void Application::portsSelected(int index)
{
    MCAM_LOG_INIT("Application::portsSelected")
    long currentIndex = 0;
    MCAM_LOGF_INFO("changed port to index=%d", index);
    bool right = index > 0;
    bool bottom = index > 1;
    long result = McammUseSensorTaps(cameraIndex, right, bottom);
    if (result != NOERR)
        MCAM_LOGF_ERROR("failed to set taps right=%d bottom=%d, result=%ld", right, bottom, result);
    result = McammCurrentUsedSensorTaps(cameraIndex, &right, &bottom);
    if (result == NOERR) {
        MCAM_LOGF_INFO("current Taps right=%d bottom=%d", right, bottom);
        if (right)
            currentIndex = 1;
        if (bottom)
            currentIndex = 2;
        ui->portComboBox->setCurrentIndex(currentIndex);
    }
}

// GUI callback
void Application::handleFullROIButton()
{
    MCAM_LOG_INIT("Application::handleFullROIButton")
    RECT rcSize;
    long result = NOERR;
    result = McammSetFrameSize(cameraIndex, NULL);
    if (result != NOERR)
        MCAM_LOGF_ERROR("failed to set max frame size, result=%ld", result);

    result = McammGetCurrentFrameSize(cameraIndex, &rcSize);
    if (result == NOERR) {
        ui->posY->setText(QString::number(rcSize.top));
        ui->posX->setText(QString::number(rcSize.left));
        ui->sizeX->setText(QString::number(rcSize.right - rcSize.left));
        ui->sizeY->setText(QString::number(rcSize.bottom - rcSize.top));
    }
    updateCompressionMode();
}

// GUI callback
void Application::handleApplyROIButton()
{
    MCAM_LOG_INIT("Application::handleApplyROIButton")
    RECT rcSize;
    long result = NOERR;
    long posX = ui->posX->displayText().toInt();
    long posY = ui->posY->displayText().toInt();
    long sizeX = ui->sizeX->displayText().toInt();
    long sizeY = ui->sizeY->displayText().toInt();
    rcSize.top = posY;
    rcSize.bottom = posY + sizeY;
    rcSize.right = posX + sizeX;
    rcSize.left = posX;
    result = McammSetFrameSize(cameraIndex, &rcSize);
    if (result != NOERR)
        MCAM_LOGF_ERROR("failed to set frame size, result=%ld", result);

    result = McammGetCurrentFrameSize(cameraIndex, &rcSize);
    if (result == NOERR) {
        ui->posY->setText(QString::number(rcSize.top));
        ui->posX->setText(QString::number(rcSize.left));
        ui->sizeX->setText(QString::number(rcSize.right - rcSize.left));
        ui->sizeY->setText(QString::number(rcSize.bottom - rcSize.top));
    }
    updateCompressionMode();
}

RECT Application::getCurrentFrameSize()
{
	MCAM_LOG_INIT("Application::getCurrentFrameSize")
    RECT rcSize = { 0,0,0,0 };
	long result = NOERR;
	result = McammGetCurrentFrameSize(cameraIndex, &rcSize);
	if (result != NOERR)
		MCAM_LOGF_ERROR("failed to set frame size, result=%ld", result);
	return rcSize;
}

// GUI callback
void Application::updateCameraGUIParamter(long cameraIndex)
{
    long result = NOERR;
    RECT rcSize;
    long currentIndex = 0;
    bool right = false;
    bool bottom = false;
    long exposureTime = 0;
    long index = 0;
    int currentCount= 0;
    long numberOfPixelClocks = 0;
    int i;

    if (cameraIndex < 0)
    	return;

    if (!thisMCamImagePtr->isContShotRunning()) {
		// update PixelClock combo box
		currentCount = ui->frequencyComboBox->count();
		  for (i=0; i < currentCount; i++)
			  ui->frequencyComboBox->removeItem(0);
		  result = McammGetNumberOfPixelClocks(cameraIndex, &numberOfPixelClocks);
		  for (i = 0; i < numberOfPixelClocks; i++) {
			  result = McammSetPixelClock(cameraIndex, i);
			  if (result == NOERR) {
				  long value=0;
				  char buffer[1024];
				  result = McammGetPixelClockValue(cameraIndex, i, &value);
				  sprintf(buffer,"%d MHz", value/1000000);
				  ui->frequencyComboBox->addItem(QString(buffer));
			  }
		  }
    }
    result = McammGetCurrentFrameSize(cameraIndex, &rcSize);
    if (result == NOERR) {
        ui->posY->setText(QString::number(rcSize.top));
        ui->posX->setText(QString::number(rcSize.left));
        ui->sizeX->setText(QString::number(rcSize.right - rcSize.left));
        ui->sizeY->setText(QString::number(rcSize.bottom - rcSize.top));
    }
    result = McammCurrentUsedSensorTaps(cameraIndex, &right, &bottom);
    if (result == NOERR) {
        if (right)
            currentIndex = 1;
        if (bottom)
            currentIndex = 2;
        ui->portComboBox->setCurrentIndex(currentIndex);
    }
    result = McammGetCurrentBinning(cameraIndex, &currentIndex);
    if (result == NOERR)
        ui->binningComboBox->setCurrentIndex(currentIndex - 1);
    result = McammGetCurrentExposure(cameraIndex, &exposureTime);
    if (result == NOERR) {
    	currentExposureValue = exposureTime / currentExposureUnit;
        ui->exposureSpinBox->setValue(currentExposureValue);
    }

    result = McammGetCurrentPixelClock(cameraIndex, &index);
    if (result == NOERR)
        ui->frequencyComboBox->setCurrentIndex(index);

    BOOL benable = FALSE;
    result = McammIsHardwareTriggerEnabled(cameraIndex, &benable);
    if (result == NOERR) {
        if (benable && (index != 2))
            ui->triggerModeComboBox->setCurrentIndex(2);
        else if ((!benable) && (index == 2))
          ui->triggerModeComboBox->setCurrentIndex(0);
        // else: default no change
    }
    BOOL binvert = FALSE;
    result = McammGetCurrentHardwareTriggerPolarity(cameraIndex, &binvert);
    if (result == NOERR)
        ui->negativePolarityCheckBox->setChecked(binvert);

    BOOL toEdge = false;
    BOOL debounce = false;
    result = McammGetCurrentHardwareTriggerMode(cameraIndex, &toEdge, &debounce);
    if (result == NOERR) {
        ui->levelTriggerCheckBox->setChecked(!toEdge);
        ui->debounceCheckBox->setChecked(debounce);
    }
    long delay=0;
    result = McammGetCurrentHardwareTriggerDelay(cameraIndex, &delay);
    if (result == NOERR)
      ui->triggerDelaySpinBox->setValue(delay);

    updateGPOGui();
}

// GUI callback from combo box 0: off 1: SW-trigger 2: HW-trigger
void Application::handleTriggerMode(int index)
{
    long result = NOERR;
    BOOL hwTrigger = FALSE;
    MCAM_LOG_INIT("Application::handleTriggerMode")
    MCAM_LOGF_INFO("set trigger to %d", index);

    // 1st switch off
    if (index != 1) {
      thisMCamImagePtr->setSoftwareTrigger(cameraIndex, false);
    }
    if (index != 2) {
        result = McammEnableHardwareTrigger(cameraIndex, FALSE);
        if (result != NOERR)
            MCAM_LOGF_ERROR("failed to set HardwareTriggerMode, result=%ld", result);
    }
    // now enable
    if (index == 1) {
        thisMCamImagePtr->setSoftwareTrigger(cameraIndex, true);
    }
    if (index == 2) {
       result = McammEnableHardwareTrigger(cameraIndex, TRUE);
       if (result != NOERR)
           MCAM_LOGF_ERROR("failed to set HardwareTriggerMode, result=%ld", result);
    }

    BOOL benable = FALSE;
    result = McammIsHardwareTriggerEnabled(cameraIndex, &benable);
    if (result == NOERR) {
      if (benable && (index != 2))
        ui->triggerModeComboBox->setCurrentIndex(2);
      else if ((!benable) && (index == 2))
        ui->triggerModeComboBox->setCurrentIndex(0);
      // else nothing to correct
    }
}

bool Application::isTriggerEnabled() {
    return ui->triggerModeComboBox->currentIndex() != 0;
}

// GUI callback
void Application::negativePolarityChecked(bool enabled)
{
    MCAM_LOG_INIT("Application::negativePolarityChecked")
    BOOL invert = FALSE;

    MCAM_LOGF_INFO("set negativePolarityChecked to %d", enabled);
    if (enabled)
        invert = TRUE;

    long result = McammSetHardwareTriggerPolarity(cameraIndex,invert);
    if (result != NOERR) {
          MCAM_LOGF_ERROR("failed to set HardwareTriggerPolarity, result=%ld", result);
    }
    BOOL binvert = FALSE;
    result = McammGetCurrentHardwareTriggerPolarity(cameraIndex, &binvert);
    if (result == NOERR)
        ui->negativePolarityCheckBox->setChecked(binvert);
}

void Application::levelTriggerAndDebounceChecked(bool enabled)
{
    MCAM_LOG_INIT("Application::levelTriggerAndDebounce")
    long result = NOERR;
    BOOL toEdge = false;
    BOOL debounce = false;

    if (!ui->levelTriggerCheckBox->checkState())
        toEdge = true;
    if (ui->debounceCheckBox->checkState())
        debounce = true;
    MCAM_LOGF_INFO("set levelTriggerAndDebounceChecked set: level=%d debounce=%d", toEdge, debounce);
    result = McammSetHardwareTriggerMode(cameraIndex, toEdge, debounce);
    if (result != NOERR) {
        MCAM_LOGF_ERROR("failed to set HardwareTriggerMode, result=%ld", result);
    }
    toEdge = false;
    debounce = false;
    result = McammGetCurrentHardwareTriggerMode(cameraIndex, &toEdge, &debounce);
    if (result == NOERR) {
      ui->levelTriggerCheckBox->setChecked(!toEdge);
      ui->debounceCheckBox->setChecked(debounce);
    }
}

void Application::spinBoxTriggerDelayValueChanged(int value) {
    MCAM_LOG_INIT("Application::spinBoxTriggerDelayValueChanged")

    long result = NOERR;
    MCAM_LOGF_INFO("set to to %d", value);
    long delay = value;
    result = McammSetHardwareTriggerDelay(cameraIndex, delay);
    if (result != NOERR) {
        MCAM_LOGF_ERROR("failed to set to %ld, result=%ld", delay, result);
    }
    delay=0;
    result = McammGetCurrentHardwareTriggerDelay(cameraIndex, &delay);
    if (result == NOERR)
        ui->triggerDelaySpinBox->setValue(delay);
}

// update GPO GUI
void Application::updateGPOGui(){
    MCAM_LOG_INIT("Application::updateGPOGui")
    int gpoIndex = ui->gpoIndexComboBox->currentIndex();
    long result = NOERR;
    MCammGPOSource gpoSource = mcammGPOOff;

    result = McammGetGPOSource(cameraIndex, gpoIndex, &gpoSource);
    if (result == NOERR) {
        if (gpoSource == mcammGPOOff)
            ui->gpoSrcComboBox->setCurrentIndex(0);
        else if (gpoSource == mcammGPOTriggered)
            ui->gpoSrcComboBox->setCurrentIndex(1);
        else if (gpoSource == mcammGPOExposure)
            ui->gpoSrcComboBox->setCurrentIndex(2);
        else if (gpoSource == mcammGPOReadout)
            ui->gpoSrcComboBox->setCurrentIndex(3);
        else if (gpoSource == mcammGPOSyncTriggerReady)
            ui->gpoSrcComboBox->setCurrentIndex(4);
        else if (gpoSource == mcammGPOAsyncTriggerReady)
            ui->gpoSrcComboBox->setCurrentIndex(5);
        else
            ui->gpoSrcComboBox->setCurrentIndex(0);
    } else
        MCAM_LOGF_ERROR("McammGetGPOSource gpoIndex=%d failed to get, result=%ld", gpoIndex, result);
    long delay = 0;
    long pulseWidth = 0;
    bool inverted = false;
    result = McammGetGPOSettings(cameraIndex, gpoIndex, &delay, &pulseWidth, &inverted);
    if (result == NOERR) {
        ui->gpoInvertedCheckBox->setChecked(inverted);
        ui->gpoPulseSpinBox->setValue(pulseWidth);
        ui->gpoDelaySpinBox->setValue(delay);
    } else
      MCAM_LOGF_ERROR("McammGetGPOSettings gpoIndex=%d failed to get, result=%ld", gpoIndex, result);
}

// GUI Callback
// select GPO 0 ..2 changed
void Application::gpoIndexSelectedComboBox(int value){
    // load settings accordingly
    updateGPOGui();
}

// GUI Callback
void Application::gpoSrcSelectedComboBox(int value){
    MCAM_LOG_INIT("Application::gpoSrcSelectedComboBox")
    int gpoIndex = ui->gpoIndexComboBox->currentIndex();
    long result = NOERR;
    MCammGPOSource source = mcammGPOOff;

    MCAM_LOGF_INFO("gpoIndex=%d set to %d", gpoIndex, value);
    switch (ui->gpoSrcComboBox->currentIndex()) {
    case 0:
    default:
      source = mcammGPOOff;
      break;
    case 1:
      source = mcammGPOTriggered;
      break;
    case 2:
      source = mcammGPOExposure;
      break;
    case 3:
      source = mcammGPOReadout;
      break;
    case 4:
      source = mcammGPOSyncTriggerReady;
      break;
    case 5:
      source = mcammGPOAsyncTriggerReady;
      break;
    }
    result = McammSetGPOSource(cameraIndex, gpoIndex, source);
    if (result != NOERR)
        MCAM_LOGF_ERROR("McammSetGPOSource gpoIndex=%d failed to set, result=%ld", gpoIndex, result);

    updateGPOGui();
}

void Application::setGPOParameter() {
  MCAM_LOG_INIT("Application::gpoSrcSelectedComboBox")
  int gpoIndex = ui->gpoIndexComboBox->currentIndex();
  long result = NOERR;
  long delay = 0;
  long pulseWidth = 0;
  bool invert = false;

  invert = ui->gpoInvertedCheckBox->checkState();
  pulseWidth = ui->gpoPulseSpinBox->value();
  delay = ui->gpoDelaySpinBox->value();

  result = McammSetGPOSettings(cameraIndex, gpoIndex, delay, pulseWidth, invert);
  if (result != NOERR)
      MCAM_LOGF_ERROR("McammSetGPOSettings gpoIndex=%d failed to set, result=%ld", gpoIndex, result);
  updateGPOGui();
}

// GUI Callback
void Application::gpoInvertedChecked(bool enabled){
  setGPOParameter();
}

// GUI Callback
void Application::gpoPulseSelectedComboBox(int value){
  setGPOParameter();
}

// GUI Callback
void Application::gpoDepaySelectedComboBox(int value){
  setGPOParameter();
}

void Application::noDiscardModeChecked(bool enabled){
	BOOL benable = (BOOL) !enabled;
	McammSetImageDiscardMode(cameraIndex, benable);
	McammGetImageDiscardMode(cameraIndex, &benable);
	//ui->noDiscardModeCheckBox->setChecked(!benable);
}

void Application::HDRModeChecked(bool enabled){
	BOOL benable = (BOOL) enabled;
	McammSetHDRMode(cameraIndex, benable);
	McammGetHDRMode(cameraIndex, &benable);
	ui->HDRModeCheckBox->setChecked(benable);
	if (benable)
		McammSetImageDiscardMode(cameraIndex, false);
	else
		McammSetImageDiscardMode(cameraIndex, true);
	updateCameraGUIParamter(cameraIndex);
}

void Application::HighRateModeChecked(bool enabled){
	BOOL benable = (BOOL) enabled;
	McammEnableHighImageRateMode(cameraIndex, benable);
	McammIsHighImageRateModeEnabled(cameraIndex, &benable);
	ui->HighRateCheckBox->setChecked(benable);
	if (benable)
		McammEnableHighImageRateMode(cameraIndex, true);
	else
		McammEnableHighImageRateMode(cameraIndex, false);
	updateCameraGUIParamter(cameraIndex);
}

void Application::updateHighRateModeChecked() {
	BOOL benable;
	McammIsHighImageRateModeEnabled(cameraIndex, &benable);
	ui->HighRateCheckBox->setChecked(benable);
}


void Application::BufferChecked(bool enabled){

  McammSetCameraBuffering(cameraIndex, enabled);
  McammGetCurrentCameraBuffering(cameraIndex, &enabled);

 // cameraBufferEnabled = enabled;

  ui->BufferCheckBox->setChecked(enabled);
  McammSetCameraBuffering(cameraIndex, enabled);
  updateCameraGUIParamter(cameraIndex);
}

bool Application::isBufferEnabledCurrentCamera() {
  return cameraBufferEnabled;
}

void Application::updateBufferChecked() {
  bool enabled;
  McammGetCurrentCameraBuffering(cameraIndex, &enabled);
  ui->BufferCheckBox->setChecked(enabled);
}

void Application::updateTileAdjustmentSetting()
{
    MCammTileAdjustmentMode adjMode = mcammTileAdjustmentOff;
    long result = McammGetTileAdjustmentMode(cameraIndex, &adjMode);

    if (result == NOERR) {
        switch(adjMode) {
            case mcammTileAdjustmentOff:
                ui->actionTileAdjustmentOff->setChecked(true);
                break;
            case mcammTileAdjustmentLinear:
                ui->actionTileAdjustmentLinear->setChecked(true);
                break;
            default:
                break;
        }
    }
}

void Application::updateLineFlickerSuppresionSetting()
{
    MCammLineFlickerSuppressionMode adjMode = mcammLineFlickerSuppressionOff;
    long result = McammGetLineFlickerSuppressionMode(cameraIndex, &adjMode);

    if (result == NOERR) {
        switch(adjMode) {
            case mcammLineFlickerSuppressionOff:
                ui->actionLineFlickerSuppressionOff->setChecked(true);
                break;
            case mcammLineFlickerSuppressionLinear:
                ui->actionLineFlickerSuppressionLinear->setChecked(true);
                break;
            case mcammLineFlickerSuppressionBiLinear:
                ui->actionLineFlickerSuppressionBilinear->setChecked(true);
                break;
            default:
                break;
        }
    }
}
// called on Combo Box change
void Application::handleCompressionModeChange(int value) {
	BOOL isFaster = FALSE;
	BOOL bufferingIsFaster = FALSE;

	switch (value) {
	case 0:
		MCammIs8BitCompressionFaster(cameraIndex, &isFaster);
		McammEnable8bitCompression(cameraIndex, isFaster);
		break;
	case 1:
		McammEnable8bitCompression(cameraIndex, true);
		break;
	case 2:
		McammEnable8bitCompression(cameraIndex, false);
		break;
	default:
		break;
	}
	updateCompressionMode();
	MCammIsImageBufferingFaster(cameraIndex, &bufferingIsFaster);
	McammSetCameraBuffering(cameraIndex, bufferingIsFaster);
    ui->BufferCheckBox->setChecked(bufferingIsFaster);
}

// called after conditions change
void Application::updateCompressionMode() {
	BOOL compressionEnabled = FALSE;
	long currentIndex = ui->compressionComboBox->currentIndex();
	if (currentIndex != 0) {
		// not auto
		Mcammis8bitCompressionEnabled(cameraIndex, &compressionEnabled);
		if (compressionEnabled)
			ui->compressionComboBox->setCurrentIndex(1);  // on
		else
			ui->compressionComboBox->setCurrentIndex(2);  // off
	} else {
		// auto mode
		BOOL isFaster = FALSE;
		MCammIs8BitCompressionFaster(cameraIndex, &isFaster);
		McammEnable8bitCompression(cameraIndex, isFaster);
	}
}

void Application::histogram() {
	 bool isEnabled = ui->actionHistogram->isChecked();
     thisMCamImagePtr->enableHistogram(isEnabled);
}
