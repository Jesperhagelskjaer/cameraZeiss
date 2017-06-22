/********************************************************************************
** Form generated from reading UI file 'mcam.ui'
**
** Created by: Qt User Interface Compiler version 4.8.6
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MCAM_H
#define UI_MCAM_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QActionGroup>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDockWidget>
#include <QtGui/QFormLayout>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QPushButton>
#include <QtGui/QScrollArea>
#include <QtGui/QSlider>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QStatusBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Application
{
public:
    QAction *actionOpen;
    QAction *actionSave;
    QAction *actionExit;
    QAction *actionNoDevice;
    QAction *actionSingleShot;
    QAction *actionContinuousShot;
    QAction *actionLoadSettings;
    QAction *actionStartStressTest;
    QAction *actionLowQualityDemosaicing;
    QAction *actionMediumQualityDemosaicing;
    QAction *actionHighQualityDemosaicing;
    QAction *actionTileAdjustmentOff;
    QAction *actionTileAdjustmentLinear;
    QAction *actionTileAdjustmentBilinear;
    QAction *actionLineFlickerSuppressionOff;
    QAction *actionLineFlickerSuppressionLinear;
    QAction *actionLineFlickerSuppressionBilinear;
    QAction *actionCalcBlackReference;
    QAction *actionEnableBlackReference;
    QAction *actionSaveBlackReference;
    QAction *actionRestoreBlackReference;
    QAction *actionCalcWhiteReference;
    QAction *actionEnableWhiteReference;
    QAction *actionSaveWhiteReference;
    QAction *actionRestoreWhiteReference;
    QAction *sqrtImage;
    QAction *linGainImage;
    QAction *actionZoomIn;
    QAction *actionZoomOut;
    QAction *actionNormalSize;
    QAction *actionFitToWindow;
    QAction *actionEditProperties;
    QAction *actionAbout;
    QAction *actionHistogram;
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout;
    QScrollArea *scrollArea;
    QLabel *imageLabel;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuCamera;
    QMenu *menuDevice;
    QActionGroup *actionGroupDevices;
    QMenu *menuImageProcessing;
    QMenu *menuDemosaicing;
    QActionGroup *actionGroupDemosaicingQuality;
    QMenu *menuTileAdjustment;
    QActionGroup *actionGroupTileAdjustment;
    QMenu *menuLineFlickerSuppression;
    QActionGroup *actionGroupLineFlickerSuppression;
    QMenu *menuBlackReference;
    QMenu *menuWhiteReference;
    QMenu *menuView;
    QMenu *menuConfig;
    QMenu *menuHelp;
    QStatusBar *statusbar;
    QLabel *fpsStatusLabel;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents;
    QGridLayout *gridLayout_3;
    QComboBox *cameraComboBox;
    QHBoxLayout *horizontalLayout_8;
    QPushButton *genericShotButton;
    QPushButton *singleShotButton;
    QPushButton *contShotButton;
    QGroupBox *groupBox_5;
    QGridLayout *gridLayout;
    QLineEdit *sizeYalgo;
    QPushButton *applyROIButton;
    QLineEdit *sizeXalgo;
    QPushButton *fullROIButton;
    QLineEdit *sizeY;
    QLineEdit *posX;
    QLineEdit *posYalgo;
    QLabel *label_11;
    QLabel *label_17;
    QLabel *label_9;
    QComboBox *binningComboBox;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_16;
    QComboBox *compressionComboBox;
    QCheckBox *HDRModeCheckBox;
    QCheckBox *HighRateCheckBox;
    QCheckBox *BufferCheckBox;
    QComboBox *portComboBox;
    QLineEdit *sizeX;
    QLabel *label_18;
    QLineEdit *posY;
    QLineEdit *posXalgo;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_2;
    QComboBox *exposureTimeUnitComboBox;
    QSlider *exposureSlider;
    QSpinBox *exposureSpinBox;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout_4;
    QComboBox *frequencyComboBox;
    QLineEdit *costText;
    QHBoxLayout *horizontalLayout_13;
    QGroupBox *groupBox_6;
    QVBoxLayout *verticalLayout_5;
    QComboBox *triggerModeComboBox;
    QCheckBox *negativePolarityCheckBox;
    QCheckBox *levelTriggerCheckBox;
    QCheckBox *debounceCheckBox;
    QHBoxLayout *horizontalLayout_7;
    QLabel *label_10;
    QSpinBox *triggerDelaySpinBox;
    QGroupBox *groupBox_7;
    QVBoxLayout *verticalLayout_4;
    QHBoxLayout *horizontalLayout_12;
    QLabel *label_15;
    QComboBox *gpoIndexComboBox;
    QHBoxLayout *horizontalLayout_9;
    QLabel *label_12;
    QComboBox *gpoSrcComboBox;
    QCheckBox *gpoInvertedCheckBox;
    QHBoxLayout *horizontalLayout_10;
    QLabel *label_13;
    QSpinBox *gpoPulseSpinBox;
    QHBoxLayout *horizontalLayout_11;
    QLabel *label_14;
    QSpinBox *gpoDelaySpinBox;
    QGroupBox *groupBox_3;
    QFormLayout *formLayout;
    QLabel *label_19;
    QLineEdit *iterations;
    QLabel *label_20;
    QLineEdit *parents;
    QLabel *label_21;
    QLineEdit *bindings;
    QLabel *label_22;
    QLineEdit *between;
    QLabel *label_23;
    QLineEdit *intensity;
    QLabel *label_24;
    QLineEdit *delay;
    QLabel *label_25;
    QLineEdit *pause;
    QPushButton *saveUserButton;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_14;
    QLabel *label_5;
    QSlider *colorTemperaturSlider;
    QLabel *label_4;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label;
    QSlider *redSlider;
    QLabel *label_2;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_3;
    QSlider *greenSlider;
    QLabel *label_6;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_7;
    QSlider *blueSlider;
    QLabel *label_8;
    QHBoxLayout *horizontalLayout_6;
    QSpacerItem *horizontalSpacer;
    QPushButton *saveColorButton;
    QPushButton *defaultColorButton;
    QPushButton *resetColorButton;

    void setupUi(QMainWindow *Application)
    {
        if (Application->objectName().isEmpty())
            Application->setObjectName(QString::fromUtf8("Application"));
        Application->resize(1045, 1024);
        actionOpen = new QAction(Application);
        actionOpen->setObjectName(QString::fromUtf8("actionOpen"));
        actionSave = new QAction(Application);
        actionSave->setObjectName(QString::fromUtf8("actionSave"));
        actionSave->setEnabled(false);
        actionExit = new QAction(Application);
        actionExit->setObjectName(QString::fromUtf8("actionExit"));
        actionNoDevice = new QAction(Application);
        actionNoDevice->setObjectName(QString::fromUtf8("actionNoDevice"));
        actionNoDevice->setEnabled(false);
        actionSingleShot = new QAction(Application);
        actionSingleShot->setObjectName(QString::fromUtf8("actionSingleShot"));
        actionContinuousShot = new QAction(Application);
        actionContinuousShot->setObjectName(QString::fromUtf8("actionContinuousShot"));
        actionContinuousShot->setCheckable(true);
        actionLoadSettings = new QAction(Application);
        actionLoadSettings->setObjectName(QString::fromUtf8("actionLoadSettings"));
        actionStartStressTest = new QAction(Application);
        actionStartStressTest->setObjectName(QString::fromUtf8("actionStartStressTest"));
        actionLowQualityDemosaicing = new QAction(Application);
        actionLowQualityDemosaicing->setObjectName(QString::fromUtf8("actionLowQualityDemosaicing"));
        actionLowQualityDemosaicing->setCheckable(true);
        actionMediumQualityDemosaicing = new QAction(Application);
        actionMediumQualityDemosaicing->setObjectName(QString::fromUtf8("actionMediumQualityDemosaicing"));
        actionMediumQualityDemosaicing->setCheckable(true);
        actionHighQualityDemosaicing = new QAction(Application);
        actionHighQualityDemosaicing->setObjectName(QString::fromUtf8("actionHighQualityDemosaicing"));
        actionHighQualityDemosaicing->setCheckable(true);
        actionTileAdjustmentOff = new QAction(Application);
        actionTileAdjustmentOff->setObjectName(QString::fromUtf8("actionTileAdjustmentOff"));
        actionTileAdjustmentOff->setCheckable(true);
        actionTileAdjustmentLinear = new QAction(Application);
        actionTileAdjustmentLinear->setObjectName(QString::fromUtf8("actionTileAdjustmentLinear"));
        actionTileAdjustmentLinear->setCheckable(true);
        actionTileAdjustmentBilinear = new QAction(Application);
        actionTileAdjustmentBilinear->setObjectName(QString::fromUtf8("actionTileAdjustmentBilinear"));
        actionTileAdjustmentBilinear->setCheckable(true);
        actionLineFlickerSuppressionOff = new QAction(Application);
        actionLineFlickerSuppressionOff->setObjectName(QString::fromUtf8("actionLineFlickerSuppressionOff"));
        actionLineFlickerSuppressionOff->setCheckable(true);
        actionLineFlickerSuppressionLinear = new QAction(Application);
        actionLineFlickerSuppressionLinear->setObjectName(QString::fromUtf8("actionLineFlickerSuppressionLinear"));
        actionLineFlickerSuppressionLinear->setCheckable(true);
        actionLineFlickerSuppressionBilinear = new QAction(Application);
        actionLineFlickerSuppressionBilinear->setObjectName(QString::fromUtf8("actionLineFlickerSuppressionBilinear"));
        actionLineFlickerSuppressionBilinear->setCheckable(true);
        actionCalcBlackReference = new QAction(Application);
        actionCalcBlackReference->setObjectName(QString::fromUtf8("actionCalcBlackReference"));
        actionEnableBlackReference = new QAction(Application);
        actionEnableBlackReference->setObjectName(QString::fromUtf8("actionEnableBlackReference"));
        actionEnableBlackReference->setCheckable(true);
        actionSaveBlackReference = new QAction(Application);
        actionSaveBlackReference->setObjectName(QString::fromUtf8("actionSaveBlackReference"));
        actionRestoreBlackReference = new QAction(Application);
        actionRestoreBlackReference->setObjectName(QString::fromUtf8("actionRestoreBlackReference"));
        actionCalcWhiteReference = new QAction(Application);
        actionCalcWhiteReference->setObjectName(QString::fromUtf8("actionCalcWhiteReference"));
        actionEnableWhiteReference = new QAction(Application);
        actionEnableWhiteReference->setObjectName(QString::fromUtf8("actionEnableWhiteReference"));
        actionEnableWhiteReference->setCheckable(true);
        actionSaveWhiteReference = new QAction(Application);
        actionSaveWhiteReference->setObjectName(QString::fromUtf8("actionSaveWhiteReference"));
        actionRestoreWhiteReference = new QAction(Application);
        actionRestoreWhiteReference->setObjectName(QString::fromUtf8("actionRestoreWhiteReference"));
        sqrtImage = new QAction(Application);
        sqrtImage->setObjectName(QString::fromUtf8("sqrtImage"));
        sqrtImage->setCheckable(true);
        sqrtImage->setEnabled(true);
        linGainImage = new QAction(Application);
        linGainImage->setObjectName(QString::fromUtf8("linGainImage"));
        linGainImage->setCheckable(true);
        linGainImage->setEnabled(true);
        actionZoomIn = new QAction(Application);
        actionZoomIn->setObjectName(QString::fromUtf8("actionZoomIn"));
        actionZoomIn->setEnabled(false);
        actionZoomOut = new QAction(Application);
        actionZoomOut->setObjectName(QString::fromUtf8("actionZoomOut"));
        actionZoomOut->setEnabled(false);
        actionNormalSize = new QAction(Application);
        actionNormalSize->setObjectName(QString::fromUtf8("actionNormalSize"));
        actionNormalSize->setEnabled(false);
        actionFitToWindow = new QAction(Application);
        actionFitToWindow->setObjectName(QString::fromUtf8("actionFitToWindow"));
        actionFitToWindow->setCheckable(true);
        actionFitToWindow->setEnabled(false);
        actionEditProperties = new QAction(Application);
        actionEditProperties->setObjectName(QString::fromUtf8("actionEditProperties"));
        actionAbout = new QAction(Application);
        actionAbout->setObjectName(QString::fromUtf8("actionAbout"));
        actionHistogram = new QAction(Application);
        actionHistogram->setObjectName(QString::fromUtf8("actionHistogram"));
        actionHistogram->setCheckable(true);
        centralWidget = new QWidget(Application);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        horizontalLayout = new QHBoxLayout(centralWidget);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        scrollArea = new QScrollArea(centralWidget);
        scrollArea->setObjectName(QString::fromUtf8("scrollArea"));
        scrollArea->setEnabled(false);
        scrollArea->setFrameShape(QFrame::NoFrame);
        scrollArea->setFrameShadow(QFrame::Plain);
        scrollArea->setLineWidth(0);
        imageLabel = new QLabel();
        imageLabel->setObjectName(QString::fromUtf8("imageLabel"));
        imageLabel->setGeometry(QRect(0, 0, 6, 14));
        QSizePolicy sizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(imageLabel->sizePolicy().hasHeightForWidth());
        imageLabel->setSizePolicy(sizePolicy);
        imageLabel->setScaledContents(true);
        scrollArea->setWidget(imageLabel);

        horizontalLayout->addWidget(scrollArea);

        Application->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(Application);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1045, 21));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        menuCamera = new QMenu(menuBar);
        menuCamera->setObjectName(QString::fromUtf8("menuCamera"));
        menuDevice = new QMenu(menuCamera);
        menuDevice->setObjectName(QString::fromUtf8("menuDevice"));
        actionGroupDevices = new QActionGroup(menuDevice);
        actionGroupDevices->setObjectName(QString::fromUtf8("actionGroupDevices"));
        menuImageProcessing = new QMenu(menuBar);
        menuImageProcessing->setObjectName(QString::fromUtf8("menuImageProcessing"));
        menuDemosaicing = new QMenu(menuImageProcessing);
        menuDemosaicing->setObjectName(QString::fromUtf8("menuDemosaicing"));
        actionGroupDemosaicingQuality = new QActionGroup(menuDemosaicing);
        actionGroupDemosaicingQuality->setObjectName(QString::fromUtf8("actionGroupDemosaicingQuality"));
        menuTileAdjustment = new QMenu(menuImageProcessing);
        menuTileAdjustment->setObjectName(QString::fromUtf8("menuTileAdjustment"));
        actionGroupTileAdjustment = new QActionGroup(menuTileAdjustment);
        actionGroupTileAdjustment->setObjectName(QString::fromUtf8("actionGroupTileAdjustment"));
        menuLineFlickerSuppression = new QMenu(menuImageProcessing);
        menuLineFlickerSuppression->setObjectName(QString::fromUtf8("menuLineFlickerSuppression"));
        actionGroupLineFlickerSuppression = new QActionGroup(menuLineFlickerSuppression);
        actionGroupLineFlickerSuppression->setObjectName(QString::fromUtf8("actionGroupLineFlickerSuppression"));
        menuBlackReference = new QMenu(menuImageProcessing);
        menuBlackReference->setObjectName(QString::fromUtf8("menuBlackReference"));
        menuWhiteReference = new QMenu(menuImageProcessing);
        menuWhiteReference->setObjectName(QString::fromUtf8("menuWhiteReference"));
        menuView = new QMenu(menuBar);
        menuView->setObjectName(QString::fromUtf8("menuView"));
        menuConfig = new QMenu(menuBar);
        menuConfig->setObjectName(QString::fromUtf8("menuConfig"));
        menuHelp = new QMenu(menuBar);
        menuHelp->setObjectName(QString::fromUtf8("menuHelp"));
        Application->setMenuBar(menuBar);
        statusbar = new QStatusBar(Application);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        fpsStatusLabel = new QLabel(statusbar);
        fpsStatusLabel->setObjectName(QString::fromUtf8("fpsStatusLabel"));
        fpsStatusLabel->setGeometry(QRect(10, 0, 300, 25));
        Application->setStatusBar(statusbar);
        dockWidget = new QDockWidget(Application);
        dockWidget->setObjectName(QString::fromUtf8("dockWidget"));
        QSizePolicy sizePolicy1(QSizePolicy::MinimumExpanding, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(dockWidget->sizePolicy().hasHeightForWidth());
        dockWidget->setSizePolicy(sizePolicy1);
        dockWidget->setAutoFillBackground(true);
        dockWidget->setFloating(true);
        dockWidget->setFeatures(QDockWidget::DockWidgetFloatable|QDockWidget::DockWidgetMovable);
        dockWidget->setAllowedAreas(Qt::LeftDockWidgetArea);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        gridLayout_3 = new QGridLayout(dockWidgetContents);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        cameraComboBox = new QComboBox(dockWidgetContents);
        cameraComboBox->setObjectName(QString::fromUtf8("cameraComboBox"));

        gridLayout_3->addWidget(cameraComboBox, 0, 0, 1, 1);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        genericShotButton = new QPushButton(dockWidgetContents);
        genericShotButton->setObjectName(QString::fromUtf8("genericShotButton"));

        horizontalLayout_8->addWidget(genericShotButton);

        singleShotButton = new QPushButton(dockWidgetContents);
        singleShotButton->setObjectName(QString::fromUtf8("singleShotButton"));

        horizontalLayout_8->addWidget(singleShotButton);

        contShotButton = new QPushButton(dockWidgetContents);
        contShotButton->setObjectName(QString::fromUtf8("contShotButton"));

        horizontalLayout_8->addWidget(contShotButton);


        gridLayout_3->addLayout(horizontalLayout_8, 1, 0, 1, 1);

        groupBox_5 = new QGroupBox(dockWidgetContents);
        groupBox_5->setObjectName(QString::fromUtf8("groupBox_5"));
        groupBox_5->setMinimumSize(QSize(0, 160));
        gridLayout = new QGridLayout(groupBox_5);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        sizeYalgo = new QLineEdit(groupBox_5);
        sizeYalgo->setObjectName(QString::fromUtf8("sizeYalgo"));

        gridLayout->addWidget(sizeYalgo, 1, 5, 1, 1);

        applyROIButton = new QPushButton(groupBox_5);
        applyROIButton->setObjectName(QString::fromUtf8("applyROIButton"));

        gridLayout->addWidget(applyROIButton, 2, 3, 1, 3);

        sizeXalgo = new QLineEdit(groupBox_5);
        sizeXalgo->setObjectName(QString::fromUtf8("sizeXalgo"));

        gridLayout->addWidget(sizeXalgo, 1, 4, 1, 1);

        fullROIButton = new QPushButton(groupBox_5);
        fullROIButton->setObjectName(QString::fromUtf8("fullROIButton"));

        gridLayout->addWidget(fullROIButton, 2, 1, 1, 2);

        sizeY = new QLineEdit(groupBox_5);
        sizeY->setObjectName(QString::fromUtf8("sizeY"));

        gridLayout->addWidget(sizeY, 1, 2, 1, 1);

        posX = new QLineEdit(groupBox_5);
        posX->setObjectName(QString::fromUtf8("posX"));

        gridLayout->addWidget(posX, 0, 1, 1, 1);

        posYalgo = new QLineEdit(groupBox_5);
        posYalgo->setObjectName(QString::fromUtf8("posYalgo"));

        gridLayout->addWidget(posYalgo, 0, 5, 1, 1);

        label_11 = new QLabel(groupBox_5);
        label_11->setObjectName(QString::fromUtf8("label_11"));

        gridLayout->addWidget(label_11, 1, 0, 1, 1);

        label_17 = new QLabel(groupBox_5);
        label_17->setObjectName(QString::fromUtf8("label_17"));

        gridLayout->addWidget(label_17, 0, 3, 1, 1);

        label_9 = new QLabel(groupBox_5);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        gridLayout->addWidget(label_9, 0, 0, 1, 1);

        binningComboBox = new QComboBox(groupBox_5);
        binningComboBox->setObjectName(QString::fromUtf8("binningComboBox"));

        gridLayout->addWidget(binningComboBox, 3, 1, 1, 2);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_16 = new QLabel(groupBox_5);
        label_16->setObjectName(QString::fromUtf8("label_16"));

        horizontalLayout_5->addWidget(label_16);

        compressionComboBox = new QComboBox(groupBox_5);
        compressionComboBox->setObjectName(QString::fromUtf8("compressionComboBox"));

        horizontalLayout_5->addWidget(compressionComboBox);

        HDRModeCheckBox = new QCheckBox(groupBox_5);
        HDRModeCheckBox->setObjectName(QString::fromUtf8("HDRModeCheckBox"));

        horizontalLayout_5->addWidget(HDRModeCheckBox);

        HighRateCheckBox = new QCheckBox(groupBox_5);
        HighRateCheckBox->setObjectName(QString::fromUtf8("HighRateCheckBox"));

        horizontalLayout_5->addWidget(HighRateCheckBox);

        BufferCheckBox = new QCheckBox(groupBox_5);
        BufferCheckBox->setObjectName(QString::fromUtf8("BufferCheckBox"));

        horizontalLayout_5->addWidget(BufferCheckBox);


        gridLayout->addLayout(horizontalLayout_5, 4, 0, 1, 6);

        portComboBox = new QComboBox(groupBox_5);
        portComboBox->setObjectName(QString::fromUtf8("portComboBox"));

        gridLayout->addWidget(portComboBox, 3, 3, 1, 3);

        sizeX = new QLineEdit(groupBox_5);
        sizeX->setObjectName(QString::fromUtf8("sizeX"));

        gridLayout->addWidget(sizeX, 1, 1, 1, 1);

        label_18 = new QLabel(groupBox_5);
        label_18->setObjectName(QString::fromUtf8("label_18"));

        gridLayout->addWidget(label_18, 1, 3, 1, 1);

        posY = new QLineEdit(groupBox_5);
        posY->setObjectName(QString::fromUtf8("posY"));

        gridLayout->addWidget(posY, 0, 2, 1, 1);

        posXalgo = new QLineEdit(groupBox_5);
        posXalgo->setObjectName(QString::fromUtf8("posXalgo"));

        gridLayout->addWidget(posXalgo, 0, 4, 1, 1);


        gridLayout_3->addWidget(groupBox_5, 2, 0, 1, 1);

        groupBox = new QGroupBox(dockWidgetContents);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setMinimumSize(QSize(0, 90));
        gridLayout_2 = new QGridLayout(groupBox);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        exposureTimeUnitComboBox = new QComboBox(groupBox);
        exposureTimeUnitComboBox->setObjectName(QString::fromUtf8("exposureTimeUnitComboBox"));

        gridLayout_2->addWidget(exposureTimeUnitComboBox, 0, 1, 1, 1);

        exposureSlider = new QSlider(groupBox);
        exposureSlider->setObjectName(QString::fromUtf8("exposureSlider"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(1);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(exposureSlider->sizePolicy().hasHeightForWidth());
        exposureSlider->setSizePolicy(sizePolicy2);
        exposureSlider->setMaximum(99999);
        exposureSlider->setOrientation(Qt::Horizontal);

        gridLayout_2->addWidget(exposureSlider, 1, 0, 1, 2);

        exposureSpinBox = new QSpinBox(groupBox);
        exposureSpinBox->setObjectName(QString::fromUtf8("exposureSpinBox"));
        exposureSpinBox->setMinimumSize(QSize(100, 0));
        exposureSpinBox->setMinimum(1);
        exposureSpinBox->setMaximum(100000);

        gridLayout_2->addWidget(exposureSpinBox, 0, 0, 1, 1);


        gridLayout_3->addWidget(groupBox, 3, 0, 1, 1);

        groupBox_4 = new QGroupBox(dockWidgetContents);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        groupBox_4->setMinimumSize(QSize(0, 50));
        gridLayout_4 = new QGridLayout(groupBox_4);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        frequencyComboBox = new QComboBox(groupBox_4);
        frequencyComboBox->setObjectName(QString::fromUtf8("frequencyComboBox"));

        gridLayout_4->addWidget(frequencyComboBox, 0, 0, 1, 1);

        costText = new QLineEdit(groupBox_4);
        costText->setObjectName(QString::fromUtf8("costText"));
        costText->setReadOnly(true);

        gridLayout_4->addWidget(costText, 0, 1, 1, 1);


        gridLayout_3->addWidget(groupBox_4, 4, 0, 1, 1);

        horizontalLayout_13 = new QHBoxLayout();
        horizontalLayout_13->setObjectName(QString::fromUtf8("horizontalLayout_13"));
        groupBox_6 = new QGroupBox(dockWidgetContents);
        groupBox_6->setObjectName(QString::fromUtf8("groupBox_6"));
        verticalLayout_5 = new QVBoxLayout(groupBox_6);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        triggerModeComboBox = new QComboBox(groupBox_6);
        triggerModeComboBox->setObjectName(QString::fromUtf8("triggerModeComboBox"));

        verticalLayout_5->addWidget(triggerModeComboBox);

        negativePolarityCheckBox = new QCheckBox(groupBox_6);
        negativePolarityCheckBox->setObjectName(QString::fromUtf8("negativePolarityCheckBox"));

        verticalLayout_5->addWidget(negativePolarityCheckBox);

        levelTriggerCheckBox = new QCheckBox(groupBox_6);
        levelTriggerCheckBox->setObjectName(QString::fromUtf8("levelTriggerCheckBox"));

        verticalLayout_5->addWidget(levelTriggerCheckBox);

        debounceCheckBox = new QCheckBox(groupBox_6);
        debounceCheckBox->setObjectName(QString::fromUtf8("debounceCheckBox"));

        verticalLayout_5->addWidget(debounceCheckBox);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        label_10 = new QLabel(groupBox_6);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        horizontalLayout_7->addWidget(label_10);

        triggerDelaySpinBox = new QSpinBox(groupBox_6);
        triggerDelaySpinBox->setObjectName(QString::fromUtf8("triggerDelaySpinBox"));
        triggerDelaySpinBox->setMaximum(16777215);

        horizontalLayout_7->addWidget(triggerDelaySpinBox);


        verticalLayout_5->addLayout(horizontalLayout_7);


        horizontalLayout_13->addWidget(groupBox_6);

        groupBox_7 = new QGroupBox(dockWidgetContents);
        groupBox_7->setObjectName(QString::fromUtf8("groupBox_7"));
        verticalLayout_4 = new QVBoxLayout(groupBox_7);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        label_15 = new QLabel(groupBox_7);
        label_15->setObjectName(QString::fromUtf8("label_15"));

        horizontalLayout_12->addWidget(label_15);

        gpoIndexComboBox = new QComboBox(groupBox_7);
        gpoIndexComboBox->setObjectName(QString::fromUtf8("gpoIndexComboBox"));

        horizontalLayout_12->addWidget(gpoIndexComboBox);


        verticalLayout_4->addLayout(horizontalLayout_12);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        label_12 = new QLabel(groupBox_7);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        horizontalLayout_9->addWidget(label_12);

        gpoSrcComboBox = new QComboBox(groupBox_7);
        gpoSrcComboBox->setObjectName(QString::fromUtf8("gpoSrcComboBox"));

        horizontalLayout_9->addWidget(gpoSrcComboBox);


        verticalLayout_4->addLayout(horizontalLayout_9);

        gpoInvertedCheckBox = new QCheckBox(groupBox_7);
        gpoInvertedCheckBox->setObjectName(QString::fromUtf8("gpoInvertedCheckBox"));

        verticalLayout_4->addWidget(gpoInvertedCheckBox);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        label_13 = new QLabel(groupBox_7);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        horizontalLayout_10->addWidget(label_13);

        gpoPulseSpinBox = new QSpinBox(groupBox_7);
        gpoPulseSpinBox->setObjectName(QString::fromUtf8("gpoPulseSpinBox"));
        gpoPulseSpinBox->setMaximum(67108863);

        horizontalLayout_10->addWidget(gpoPulseSpinBox);


        verticalLayout_4->addLayout(horizontalLayout_10);

        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        label_14 = new QLabel(groupBox_7);
        label_14->setObjectName(QString::fromUtf8("label_14"));

        horizontalLayout_11->addWidget(label_14);

        gpoDelaySpinBox = new QSpinBox(groupBox_7);
        gpoDelaySpinBox->setObjectName(QString::fromUtf8("gpoDelaySpinBox"));
        gpoDelaySpinBox->setMaximum(16777215);

        horizontalLayout_11->addWidget(gpoDelaySpinBox);


        verticalLayout_4->addLayout(horizontalLayout_11);


        horizontalLayout_13->addWidget(groupBox_7);


        gridLayout_3->addLayout(horizontalLayout_13, 5, 0, 1, 1);

        groupBox_3 = new QGroupBox(dockWidgetContents);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        formLayout = new QFormLayout(groupBox_3);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        label_19 = new QLabel(groupBox_3);
        label_19->setObjectName(QString::fromUtf8("label_19"));

        formLayout->setWidget(0, QFormLayout::LabelRole, label_19);

        iterations = new QLineEdit(groupBox_3);
        iterations->setObjectName(QString::fromUtf8("iterations"));

        formLayout->setWidget(0, QFormLayout::FieldRole, iterations);

        label_20 = new QLabel(groupBox_3);
        label_20->setObjectName(QString::fromUtf8("label_20"));

        formLayout->setWidget(1, QFormLayout::LabelRole, label_20);

        parents = new QLineEdit(groupBox_3);
        parents->setObjectName(QString::fromUtf8("parents"));

        formLayout->setWidget(1, QFormLayout::FieldRole, parents);

        label_21 = new QLabel(groupBox_3);
        label_21->setObjectName(QString::fromUtf8("label_21"));

        formLayout->setWidget(2, QFormLayout::LabelRole, label_21);

        bindings = new QLineEdit(groupBox_3);
        bindings->setObjectName(QString::fromUtf8("bindings"));

        formLayout->setWidget(2, QFormLayout::FieldRole, bindings);

        label_22 = new QLabel(groupBox_3);
        label_22->setObjectName(QString::fromUtf8("label_22"));

        formLayout->setWidget(3, QFormLayout::LabelRole, label_22);

        between = new QLineEdit(groupBox_3);
        between->setObjectName(QString::fromUtf8("between"));

        formLayout->setWidget(3, QFormLayout::FieldRole, between);

        label_23 = new QLabel(groupBox_3);
        label_23->setObjectName(QString::fromUtf8("label_23"));

        formLayout->setWidget(4, QFormLayout::LabelRole, label_23);

        intensity = new QLineEdit(groupBox_3);
        intensity->setObjectName(QString::fromUtf8("intensity"));

        formLayout->setWidget(4, QFormLayout::FieldRole, intensity);

        label_24 = new QLabel(groupBox_3);
        label_24->setObjectName(QString::fromUtf8("label_24"));

        formLayout->setWidget(5, QFormLayout::LabelRole, label_24);

        delay = new QLineEdit(groupBox_3);
        delay->setObjectName(QString::fromUtf8("delay"));

        formLayout->setWidget(5, QFormLayout::FieldRole, delay);

        label_25 = new QLabel(groupBox_3);
        label_25->setObjectName(QString::fromUtf8("label_25"));

        formLayout->setWidget(6, QFormLayout::LabelRole, label_25);

        pause = new QLineEdit(groupBox_3);
        pause->setObjectName(QString::fromUtf8("pause"));

        formLayout->setWidget(6, QFormLayout::FieldRole, pause);

        saveUserButton = new QPushButton(groupBox_3);
        saveUserButton->setObjectName(QString::fromUtf8("saveUserButton"));

        formLayout->setWidget(7, QFormLayout::FieldRole, saveUserButton);


        gridLayout_3->addWidget(groupBox_3, 6, 0, 1, 1);

        groupBox_2 = new QGroupBox(dockWidgetContents);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        groupBox_2->setEnabled(true);
        verticalLayout = new QVBoxLayout(groupBox_2);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout_14 = new QHBoxLayout();
        horizontalLayout_14->setSpacing(0);
        horizontalLayout_14->setObjectName(QString::fromUtf8("horizontalLayout_14"));
        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        horizontalLayout_14->addWidget(label_5);

        colorTemperaturSlider = new QSlider(groupBox_2);
        colorTemperaturSlider->setObjectName(QString::fromUtf8("colorTemperaturSlider"));
        colorTemperaturSlider->setMaximum(100);
        colorTemperaturSlider->setOrientation(Qt::Horizontal);

        horizontalLayout_14->addWidget(colorTemperaturSlider);

        label_4 = new QLabel(groupBox_2);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout_14->addWidget(label_4);


        verticalLayout->addLayout(horizontalLayout_14);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label = new QLabel(groupBox_2);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_2->addWidget(label);

        redSlider = new QSlider(groupBox_2);
        redSlider->setObjectName(QString::fromUtf8("redSlider"));
        redSlider->setMaximum(511);
        redSlider->setOrientation(Qt::Horizontal);

        horizontalLayout_2->addWidget(redSlider);

        label_2 = new QLabel(groupBox_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout_2->addWidget(label_2);


        verticalLayout->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(0);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_3->addWidget(label_3);

        greenSlider = new QSlider(groupBox_2);
        greenSlider->setObjectName(QString::fromUtf8("greenSlider"));
        greenSlider->setMaximum(511);
        greenSlider->setOrientation(Qt::Horizontal);

        horizontalLayout_3->addWidget(greenSlider);

        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout_3->addWidget(label_6);


        verticalLayout->addLayout(horizontalLayout_3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(0);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_7 = new QLabel(groupBox_2);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        horizontalLayout_4->addWidget(label_7);

        blueSlider = new QSlider(groupBox_2);
        blueSlider->setObjectName(QString::fromUtf8("blueSlider"));
        blueSlider->setMaximum(511);
        blueSlider->setOrientation(Qt::Horizontal);

        horizontalLayout_4->addWidget(blueSlider);

        label_8 = new QLabel(groupBox_2);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        label_8->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout_4->addWidget(label_8);


        verticalLayout->addLayout(horizontalLayout_4);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(0);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer);

        saveColorButton = new QPushButton(groupBox_2);
        saveColorButton->setObjectName(QString::fromUtf8("saveColorButton"));
        saveColorButton->setMinimumSize(QSize(40, 0));
        saveColorButton->setMaximumSize(QSize(40, 16777215));

        horizontalLayout_6->addWidget(saveColorButton);

        defaultColorButton = new QPushButton(groupBox_2);
        defaultColorButton->setObjectName(QString::fromUtf8("defaultColorButton"));
        defaultColorButton->setMinimumSize(QSize(0, 0));
        defaultColorButton->setMaximumSize(QSize(60, 16777215));

        horizontalLayout_6->addWidget(defaultColorButton);

        resetColorButton = new QPushButton(groupBox_2);
        resetColorButton->setObjectName(QString::fromUtf8("resetColorButton"));
        resetColorButton->setMinimumSize(QSize(0, 0));
        resetColorButton->setMaximumSize(QSize(40, 16777215));

        horizontalLayout_6->addWidget(resetColorButton);


        verticalLayout->addLayout(horizontalLayout_6);


        gridLayout_3->addWidget(groupBox_2, 7, 0, 1, 1);

        dockWidget->setWidget(dockWidgetContents);
        groupBox_5->raise();
        groupBox->raise();
        groupBox_4->raise();
        groupBox_2->raise();
        cameraComboBox->raise();
        groupBox_3->raise();
        Application->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget);
        QWidget::setTabOrder(scrollArea, cameraComboBox);
        QWidget::setTabOrder(cameraComboBox, genericShotButton);
        QWidget::setTabOrder(genericShotButton, singleShotButton);
        QWidget::setTabOrder(singleShotButton, contShotButton);
        QWidget::setTabOrder(contShotButton, posX);
        QWidget::setTabOrder(posX, posY);
        QWidget::setTabOrder(posY, posXalgo);
        QWidget::setTabOrder(posXalgo, posYalgo);
        QWidget::setTabOrder(posYalgo, sizeX);
        QWidget::setTabOrder(sizeX, sizeY);
        QWidget::setTabOrder(sizeY, sizeXalgo);
        QWidget::setTabOrder(sizeXalgo, sizeYalgo);
        QWidget::setTabOrder(sizeYalgo, fullROIButton);
        QWidget::setTabOrder(fullROIButton, applyROIButton);
        QWidget::setTabOrder(applyROIButton, binningComboBox);
        QWidget::setTabOrder(binningComboBox, portComboBox);
        QWidget::setTabOrder(portComboBox, compressionComboBox);
        QWidget::setTabOrder(compressionComboBox, HDRModeCheckBox);
        QWidget::setTabOrder(HDRModeCheckBox, HighRateCheckBox);
        QWidget::setTabOrder(HighRateCheckBox, BufferCheckBox);
        QWidget::setTabOrder(BufferCheckBox, exposureSpinBox);
        QWidget::setTabOrder(exposureSpinBox, exposureTimeUnitComboBox);
        QWidget::setTabOrder(exposureTimeUnitComboBox, exposureSlider);
        QWidget::setTabOrder(exposureSlider, frequencyComboBox);
        QWidget::setTabOrder(frequencyComboBox, costText);
        QWidget::setTabOrder(costText, triggerModeComboBox);
        QWidget::setTabOrder(triggerModeComboBox, negativePolarityCheckBox);
        QWidget::setTabOrder(negativePolarityCheckBox, levelTriggerCheckBox);
        QWidget::setTabOrder(levelTriggerCheckBox, debounceCheckBox);
        QWidget::setTabOrder(debounceCheckBox, triggerDelaySpinBox);
        QWidget::setTabOrder(triggerDelaySpinBox, gpoIndexComboBox);
        QWidget::setTabOrder(gpoIndexComboBox, gpoSrcComboBox);
        QWidget::setTabOrder(gpoSrcComboBox, gpoInvertedCheckBox);
        QWidget::setTabOrder(gpoInvertedCheckBox, gpoPulseSpinBox);
        QWidget::setTabOrder(gpoPulseSpinBox, gpoDelaySpinBox);
        QWidget::setTabOrder(gpoDelaySpinBox, iterations);
        QWidget::setTabOrder(iterations, parents);
        QWidget::setTabOrder(parents, bindings);
        QWidget::setTabOrder(bindings, between);
        QWidget::setTabOrder(between, intensity);
        QWidget::setTabOrder(intensity, delay);
        QWidget::setTabOrder(delay, pause);
        QWidget::setTabOrder(pause, colorTemperaturSlider);
        QWidget::setTabOrder(colorTemperaturSlider, redSlider);
        QWidget::setTabOrder(redSlider, greenSlider);
        QWidget::setTabOrder(greenSlider, blueSlider);
        QWidget::setTabOrder(blueSlider, saveColorButton);
        QWidget::setTabOrder(saveColorButton, defaultColorButton);
        QWidget::setTabOrder(defaultColorButton, resetColorButton);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuCamera->menuAction());
        menuBar->addAction(menuImageProcessing->menuAction());
        menuBar->addAction(menuView->menuAction());
        menuBar->addAction(menuConfig->menuAction());
        menuBar->addAction(menuHelp->menuAction());
        menuFile->addAction(actionOpen);
        menuFile->addAction(actionSave);
        menuFile->addSeparator();
        menuFile->addAction(actionExit);
        menuCamera->addAction(menuDevice->menuAction());
        menuCamera->addAction(actionSingleShot);
        menuCamera->addAction(actionContinuousShot);
        menuCamera->addAction(actionLoadSettings);
        menuCamera->addAction(actionStartStressTest);
        menuDevice->addSeparator();
        menuDevice->addAction(actionNoDevice);
        menuImageProcessing->addAction(menuDemosaicing->menuAction());
        menuImageProcessing->addAction(menuTileAdjustment->menuAction());
        menuImageProcessing->addAction(menuLineFlickerSuppression->menuAction());
        menuImageProcessing->addAction(menuBlackReference->menuAction());
        menuImageProcessing->addAction(menuWhiteReference->menuAction());
        menuDemosaicing->addAction(actionLowQualityDemosaicing);
        menuDemosaicing->addAction(actionMediumQualityDemosaicing);
        menuDemosaicing->addAction(actionHighQualityDemosaicing);
        actionGroupDemosaicingQuality->addAction(actionLowQualityDemosaicing);
        actionGroupDemosaicingQuality->addAction(actionMediumQualityDemosaicing);
        actionGroupDemosaicingQuality->addAction(actionHighQualityDemosaicing);
        menuTileAdjustment->addAction(actionTileAdjustmentOff);
        menuTileAdjustment->addAction(actionTileAdjustmentLinear);
        menuTileAdjustment->addAction(actionTileAdjustmentBilinear);
        actionGroupTileAdjustment->addAction(actionTileAdjustmentOff);
        actionGroupTileAdjustment->addAction(actionTileAdjustmentLinear);
        actionGroupTileAdjustment->addAction(actionTileAdjustmentBilinear);
        menuLineFlickerSuppression->addAction(actionLineFlickerSuppressionOff);
        menuLineFlickerSuppression->addAction(actionLineFlickerSuppressionLinear);
        menuLineFlickerSuppression->addAction(actionLineFlickerSuppressionBilinear);
        actionGroupLineFlickerSuppression->addAction(actionLineFlickerSuppressionOff);
        actionGroupLineFlickerSuppression->addAction(actionLineFlickerSuppressionLinear);
        actionGroupLineFlickerSuppression->addAction(actionLineFlickerSuppressionBilinear);
        menuBlackReference->addAction(actionCalcBlackReference);
        menuBlackReference->addAction(actionEnableBlackReference);
        menuBlackReference->addAction(actionSaveBlackReference);
        menuBlackReference->addAction(actionRestoreBlackReference);
        menuWhiteReference->addAction(actionCalcWhiteReference);
        menuWhiteReference->addAction(actionEnableWhiteReference);
        menuWhiteReference->addAction(actionSaveWhiteReference);
        menuWhiteReference->addAction(actionRestoreWhiteReference);
        menuView->addAction(linGainImage);
        menuView->addSeparator();
        menuView->addAction(actionZoomIn);
        menuView->addAction(actionZoomOut);
        menuView->addAction(actionNormalSize);
        menuView->addSeparator();
        menuView->addAction(actionFitToWindow);
        menuView->addSeparator();
        menuView->addAction(actionHistogram);
        menuConfig->addAction(actionEditProperties);
        menuHelp->addAction(actionAbout);

        retranslateUi(Application);
        QObject::connect(actionOpen, SIGNAL(triggered()), Application, SLOT(open()));
        QObject::connect(actionSave, SIGNAL(triggered()), Application, SLOT(save()));
        QObject::connect(actionExit, SIGNAL(triggered()), Application, SLOT(exit()));
        QObject::connect(actionSingleShot, SIGNAL(triggered()), Application, SLOT(doSingleShot()));
        QObject::connect(actionContinuousShot, SIGNAL(toggled(bool)), Application, SLOT(doContinuousShot(bool)));
        QObject::connect(actionLoadSettings, SIGNAL(triggered()), Application, SLOT(loadSettings()));
        QObject::connect(actionStartStressTest, SIGNAL(triggered()), Application, SLOT(startStressTest()));
        QObject::connect(actionLowQualityDemosaicing, SIGNAL(toggled(bool)), Application, SLOT(doLowQualityDemosaicing(bool)));
        QObject::connect(actionMediumQualityDemosaicing, SIGNAL(toggled(bool)), Application, SLOT(doMediumQualityDemosaicing(bool)));
        QObject::connect(actionHighQualityDemosaicing, SIGNAL(toggled(bool)), Application, SLOT(doHighQualityDemosaicing(bool)));
        QObject::connect(actionTileAdjustmentOff, SIGNAL(toggled(bool)), Application, SLOT(doTileAdjustmentOff(bool)));
        QObject::connect(actionTileAdjustmentLinear, SIGNAL(toggled(bool)), Application, SLOT(doTileAdjustmentLinear(bool)));
        QObject::connect(actionTileAdjustmentBilinear, SIGNAL(toggled(bool)), Application, SLOT(doTileAdjustmentBilinear(bool)));
        QObject::connect(actionLineFlickerSuppressionOff, SIGNAL(toggled(bool)), Application, SLOT(doLineFlickerSuppressionOff(bool)));
        QObject::connect(actionLineFlickerSuppressionLinear, SIGNAL(toggled(bool)), Application, SLOT(doLineFlickerSuppressionLinear(bool)));
        QObject::connect(actionLineFlickerSuppressionBilinear, SIGNAL(toggled(bool)), Application, SLOT(doLineFlickerSuppressionBilinear(bool)));
        QObject::connect(actionCalcBlackReference, SIGNAL(triggered()), Application, SLOT(calcBlackReference()));
        QObject::connect(actionEnableBlackReference, SIGNAL(toggled(bool)), Application, SLOT(doBlackReference(bool)));
        QObject::connect(actionSaveBlackReference, SIGNAL(triggered()), Application, SLOT(saveBlackReference()));
        QObject::connect(actionRestoreBlackReference, SIGNAL(triggered()), Application, SLOT(restoreBlackReference()));
        QObject::connect(actionCalcWhiteReference, SIGNAL(triggered()), Application, SLOT(calcWhiteReference()));
        QObject::connect(actionEnableWhiteReference, SIGNAL(toggled(bool)), Application, SLOT(doWhiteReference(bool)));
        QObject::connect(actionSaveWhiteReference, SIGNAL(triggered()), Application, SLOT(saveWhiteReference()));
        QObject::connect(actionRestoreWhiteReference, SIGNAL(triggered()), Application, SLOT(restoreWhiteReference()));
        QObject::connect(linGainImage, SIGNAL(toggled(bool)), Application, SLOT(doLinGainImage(bool)));
        QObject::connect(actionZoomIn, SIGNAL(triggered()), Application, SLOT(zoomIn()));
        QObject::connect(actionZoomOut, SIGNAL(triggered()), Application, SLOT(zoomOut()));
        QObject::connect(actionNormalSize, SIGNAL(triggered()), Application, SLOT(normalSize()));
        QObject::connect(actionFitToWindow, SIGNAL(triggered()), Application, SLOT(fitToWindow()));
        QObject::connect(actionEditProperties, SIGNAL(triggered()), Application, SLOT(editProperties()));
        QObject::connect(actionAbout, SIGNAL(triggered()), Application, SLOT(about()));
        QObject::connect(actionHistogram, SIGNAL(triggered()), Application, SLOT(histogram()));

        QMetaObject::connectSlotsByName(Application);
    } // setupUi

    void retranslateUi(QMainWindow *Application)
    {
        Application->setWindowTitle(QApplication::translate("Application", "MCam", 0, QApplication::UnicodeUTF8));
        actionOpen->setText(QApplication::translate("Application", "Open ...", 0, QApplication::UnicodeUTF8));
        actionOpen->setShortcut(QApplication::translate("Application", "Ctrl+O", 0, QApplication::UnicodeUTF8));
        actionSave->setText(QApplication::translate("Application", "Save ...", 0, QApplication::UnicodeUTF8));
        actionSave->setShortcut(QApplication::translate("Application", "Ctrl+S", 0, QApplication::UnicodeUTF8));
        actionExit->setText(QApplication::translate("Application", "Exit", 0, QApplication::UnicodeUTF8));
        actionExit->setShortcut(QApplication::translate("Application", "Ctrl+Q", 0, QApplication::UnicodeUTF8));
        actionNoDevice->setText(QApplication::translate("Application", "No Device", 0, QApplication::UnicodeUTF8));
        actionSingleShot->setText(QApplication::translate("Application", "SingleShot", 0, QApplication::UnicodeUTF8));
        actionSingleShot->setShortcut(QApplication::translate("Application", "F6", 0, QApplication::UnicodeUTF8));
        actionContinuousShot->setText(QApplication::translate("Application", "ContinuousShot", 0, QApplication::UnicodeUTF8));
        actionContinuousShot->setShortcut(QApplication::translate("Application", "F7", 0, QApplication::UnicodeUTF8));
        actionLoadSettings->setText(QApplication::translate("Application", "Load Settings", 0, QApplication::UnicodeUTF8));
        actionLoadSettings->setShortcut(QApplication::translate("Application", "F8", 0, QApplication::UnicodeUTF8));
        actionStartStressTest->setText(QApplication::translate("Application", "Start Stress Test", 0, QApplication::UnicodeUTF8));
        actionStartStressTest->setShortcut(QApplication::translate("Application", "F9", 0, QApplication::UnicodeUTF8));
        actionLowQualityDemosaicing->setText(QApplication::translate("Application", "Low Quality", 0, QApplication::UnicodeUTF8));
        actionMediumQualityDemosaicing->setText(QApplication::translate("Application", "Medium Quality", 0, QApplication::UnicodeUTF8));
        actionHighQualityDemosaicing->setText(QApplication::translate("Application", "High Quality", 0, QApplication::UnicodeUTF8));
        actionTileAdjustmentOff->setText(QApplication::translate("Application", "Off", 0, QApplication::UnicodeUTF8));
        actionTileAdjustmentLinear->setText(QApplication::translate("Application", "Linear", 0, QApplication::UnicodeUTF8));
        actionTileAdjustmentBilinear->setText(QApplication::translate("Application", "Bilinear", 0, QApplication::UnicodeUTF8));
        actionLineFlickerSuppressionOff->setText(QApplication::translate("Application", "Off", 0, QApplication::UnicodeUTF8));
        actionLineFlickerSuppressionLinear->setText(QApplication::translate("Application", "Linear", 0, QApplication::UnicodeUTF8));
        actionLineFlickerSuppressionBilinear->setText(QApplication::translate("Application", "Bilinear", 0, QApplication::UnicodeUTF8));
        actionCalcBlackReference->setText(QApplication::translate("Application", "Calculate Black Reference", 0, QApplication::UnicodeUTF8));
        actionEnableBlackReference->setText(QApplication::translate("Application", "Enable Black Reference", 0, QApplication::UnicodeUTF8));
        actionSaveBlackReference->setText(QApplication::translate("Application", "Save Black Reference", 0, QApplication::UnicodeUTF8));
        actionRestoreBlackReference->setText(QApplication::translate("Application", "Restore Black Reference", 0, QApplication::UnicodeUTF8));
        actionCalcWhiteReference->setText(QApplication::translate("Application", "Calculate White Reference", 0, QApplication::UnicodeUTF8));
        actionEnableWhiteReference->setText(QApplication::translate("Application", "Enable White Reference", 0, QApplication::UnicodeUTF8));
        actionSaveWhiteReference->setText(QApplication::translate("Application", "Save White Reference", 0, QApplication::UnicodeUTF8));
        actionRestoreWhiteReference->setText(QApplication::translate("Application", "Restore White Reference", 0, QApplication::UnicodeUTF8));
        sqrtImage->setText(QApplication::translate("Application", "Square Root", 0, QApplication::UnicodeUTF8));
        sqrtImage->setShortcut(QApplication::translate("Application", "Ctrl+Shift+S", 0, QApplication::UnicodeUTF8));
        linGainImage->setText(QApplication::translate("Application", "High Gain", 0, QApplication::UnicodeUTF8));
        linGainImage->setShortcut(QApplication::translate("Application", "Ctrl+Shift+G", 0, QApplication::UnicodeUTF8));
        actionZoomIn->setText(QApplication::translate("Application", "Zoom In (25%)", 0, QApplication::UnicodeUTF8));
        actionZoomIn->setShortcut(QApplication::translate("Application", "Ctrl++", 0, QApplication::UnicodeUTF8));
        actionZoomOut->setText(QApplication::translate("Application", "Zoom Out (25%)", 0, QApplication::UnicodeUTF8));
        actionZoomOut->setShortcut(QApplication::translate("Application", "Ctrl+-", 0, QApplication::UnicodeUTF8));
        actionNormalSize->setText(QApplication::translate("Application", "Normal Size", 0, QApplication::UnicodeUTF8));
        actionNormalSize->setShortcut(QApplication::translate("Application", "Ctrl+R", 0, QApplication::UnicodeUTF8));
        actionFitToWindow->setText(QApplication::translate("Application", "Fit To Window", 0, QApplication::UnicodeUTF8));
        actionFitToWindow->setShortcut(QApplication::translate("Application", "Ctrl+F", 0, QApplication::UnicodeUTF8));
        actionEditProperties->setText(QApplication::translate("Application", "Configure Axiocam DLL Properties", 0, QApplication::UnicodeUTF8));
        actionAbout->setText(QApplication::translate("Application", "About", 0, QApplication::UnicodeUTF8));
        actionAbout->setShortcut(QApplication::translate("Application", "Ctrl+A", 0, QApplication::UnicodeUTF8));
        actionHistogram->setText(QApplication::translate("Application", "Histogram", 0, QApplication::UnicodeUTF8));
        actionHistogram->setShortcut(QApplication::translate("Application", "Ctrl+H", 0, QApplication::UnicodeUTF8));
        menuFile->setTitle(QApplication::translate("Application", "File", 0, QApplication::UnicodeUTF8));
        menuCamera->setTitle(QApplication::translate("Application", "Camera", 0, QApplication::UnicodeUTF8));
        menuDevice->setTitle(QApplication::translate("Application", "Device", 0, QApplication::UnicodeUTF8));
        menuImageProcessing->setTitle(QApplication::translate("Application", "Image Processing", 0, QApplication::UnicodeUTF8));
        menuDemosaicing->setTitle(QApplication::translate("Application", "Image Quality", 0, QApplication::UnicodeUTF8));
        menuTileAdjustment->setTitle(QApplication::translate("Application", "Tile adjustment", 0, QApplication::UnicodeUTF8));
        menuLineFlickerSuppression->setTitle(QApplication::translate("Application", "Line flicker suppresion", 0, QApplication::UnicodeUTF8));
        menuBlackReference->setTitle(QApplication::translate("Application", "Black Reference", 0, QApplication::UnicodeUTF8));
        menuWhiteReference->setTitle(QApplication::translate("Application", "White Reference", 0, QApplication::UnicodeUTF8));
        menuView->setTitle(QApplication::translate("Application", "View", 0, QApplication::UnicodeUTF8));
        menuConfig->setTitle(QApplication::translate("Application", "Configuration", 0, QApplication::UnicodeUTF8));
        menuHelp->setTitle(QApplication::translate("Application", "Help", 0, QApplication::UnicodeUTF8));
        fpsStatusLabel->setText(QString());
        dockWidget->setWindowTitle(QApplication::translate("Application", " MCam Functions", 0, QApplication::UnicodeUTF8));
        genericShotButton->setText(QApplication::translate("Application", "GenericShot", 0, QApplication::UnicodeUTF8));
        singleShotButton->setText(QApplication::translate("Application", "SingleShot", 0, QApplication::UnicodeUTF8));
        contShotButton->setText(QApplication::translate("Application", "ContShot", 0, QApplication::UnicodeUTF8));
        groupBox_5->setTitle(QApplication::translate("Application", "Acquisition Parameter", 0, QApplication::UnicodeUTF8));
        applyROIButton->setText(QApplication::translate("Application", "Apply", 0, QApplication::UnicodeUTF8));
        fullROIButton->setText(QApplication::translate("Application", "Full", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("Application", "Size", 0, QApplication::UnicodeUTF8));
        label_17->setText(QApplication::translate("Application", "cPos", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("Application", "Pos", 0, QApplication::UnicodeUTF8));
        binningComboBox->clear();
        binningComboBox->insertItems(0, QStringList()
         << QApplication::translate("Application", "Binning 1x1", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "Binning 2x2", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "Binning 3x3", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "Binning 4x4", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "Binning 5x5", 0, QApplication::UnicodeUTF8)
        );
        label_16->setText(QApplication::translate("Application", "Compress", 0, QApplication::UnicodeUTF8));
        compressionComboBox->clear();
        compressionComboBox->insertItems(0, QStringList()
         << QApplication::translate("Application", "Auto", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "On", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "Off", 0, QApplication::UnicodeUTF8)
        );
        HDRModeCheckBox->setText(QApplication::translate("Application", "HDR", 0, QApplication::UnicodeUTF8));
        HighRateCheckBox->setText(QApplication::translate("Application", "HiRate", 0, QApplication::UnicodeUTF8));
        BufferCheckBox->setText(QApplication::translate("Application", "Buffer", 0, QApplication::UnicodeUTF8));
        portComboBox->clear();
        portComboBox->insertItems(0, QStringList()
         << QApplication::translate("Application", "Single Port", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "Dual Port", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "Quad Port", 0, QApplication::UnicodeUTF8)
        );
        label_18->setText(QApplication::translate("Application", "cSize", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("Application", "Exposure Time", 0, QApplication::UnicodeUTF8));
        exposureTimeUnitComboBox->clear();
        exposureTimeUnitComboBox->insertItems(0, QStringList()
         << QApplication::translate("Application", "us", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "ms", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "s", 0, QApplication::UnicodeUTF8)
        );
        groupBox_4->setTitle(QApplication::translate("Application", "Pixel Frequency                           Cost (cPos, cSize)", 0, QApplication::UnicodeUTF8));
        frequencyComboBox->clear();
        frequencyComboBox->insertItems(0, QStringList()
         << QApplication::translate("Application", "13 MHz", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "39 MHz", 0, QApplication::UnicodeUTF8)
        );
        groupBox_6->setTitle(QApplication::translate("Application", "Trigger", 0, QApplication::UnicodeUTF8));
        triggerModeComboBox->clear();
        triggerModeComboBox->insertItems(0, QStringList()
         << QApplication::translate("Application", "Off", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "SW-Trigger", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "HW-Trigger", 0, QApplication::UnicodeUTF8)
        );
        negativePolarityCheckBox->setText(QApplication::translate("Application", "Neg. Polarity", 0, QApplication::UnicodeUTF8));
        levelTriggerCheckBox->setText(QApplication::translate("Application", "Level Trigger", 0, QApplication::UnicodeUTF8));
        debounceCheckBox->setText(QApplication::translate("Application", "Debounce", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("Application", "Delay", 0, QApplication::UnicodeUTF8));
        groupBox_7->setTitle(QApplication::translate("Application", "GPO", 0, QApplication::UnicodeUTF8));
        label_15->setText(QApplication::translate("Application", "Index", 0, QApplication::UnicodeUTF8));
        gpoIndexComboBox->clear();
        gpoIndexComboBox->insertItems(0, QStringList()
         << QApplication::translate("Application", "0", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "1", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "2", 0, QApplication::UnicodeUTF8)
        );
        label_12->setText(QApplication::translate("Application", "Src", 0, QApplication::UnicodeUTF8));
        gpoSrcComboBox->clear();
        gpoSrcComboBox->insertItems(0, QStringList()
         << QApplication::translate("Application", "Off", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "Triggered", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "Exposure", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "Readout", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "SyncTR", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("Application", "AsyncTR", 0, QApplication::UnicodeUTF8)
        );
        gpoInvertedCheckBox->setText(QApplication::translate("Application", "Inverted", 0, QApplication::UnicodeUTF8));
        label_13->setText(QApplication::translate("Application", "Pulse", 0, QApplication::UnicodeUTF8));
        label_14->setText(QApplication::translate("Application", "Delay", 0, QApplication::UnicodeUTF8));
        groupBox_3->setTitle(QApplication::translate("Application", "User Parameters", 0, QApplication::UnicodeUTF8));
        label_19->setText(QApplication::translate("Application", "Iterations", 0, QApplication::UnicodeUTF8));
        label_20->setText(QApplication::translate("Application", "Parents", 0, QApplication::UnicodeUTF8));
        label_21->setText(QApplication::translate("Application", "Bindings", 0, QApplication::UnicodeUTF8));
        label_22->setText(QApplication::translate("Application", "Between", 0, QApplication::UnicodeUTF8));
        label_23->setText(QApplication::translate("Application", "Intensity", 0, QApplication::UnicodeUTF8));
        label_24->setText(QApplication::translate("Application", "Delay (ms)", 0, QApplication::UnicodeUTF8));
        label_25->setText(QApplication::translate("Application", "Pause (ms)", 0, QApplication::UnicodeUTF8));
        saveUserButton->setText(QApplication::translate("Application", "Update", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("Application", "Color Settings", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("Application", "Warm", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("Application", "Cool", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("Application", "Red", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("Application", "Cyan", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("Application", "Green", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("Application", "Magenta", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("Application", "Blue", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("Application", "Yellow", 0, QApplication::UnicodeUTF8));
        saveColorButton->setText(QApplication::translate("Application", "Save", 0, QApplication::UnicodeUTF8));
        defaultColorButton->setText(QApplication::translate("Application", "Defaults", 0, QApplication::UnicodeUTF8));
        resetColorButton->setText(QApplication::translate("Application", "Reset", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class Application: public Ui_Application {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MCAM_H
