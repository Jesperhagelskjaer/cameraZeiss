If an error happens after using QT- Designer

include/ui_mcam.h: In member function ‘void Ui_Application::setupUi(QMainWindow*)’:
include/ui_mcam.h:256:29: error: ‘class QActionGroup’ has no member named ‘setGeometry’
         actionGroupDevices->setGeometry(QRect(0, 0, 100, 30));

Remove the follwing properties:       
Suche nach "<height>30" und entferne die geometry property!

-->>> nur die geometry property entfernen!

      <widget class="QActionGroup" name="actionGroupDevices" native="true">
-      <property name="geometry">
-       <rect>
-        <x>0</x>
-        <y>0</y>
-        <width>100</width>
-        <height>30</height>
-       </rect>
-      </property>
-     </widget>


@@ -101,6 +110,14 @@
       <string>Demosaicing</string>
      </property>
      <widget class="QActionGroup" name="actionGroupDemosaicingQuality" native="true">
-      <property name="geometry">
-       <rect>
-        <x>0</x>
-        <y>0</y>
-        <width>100</width>
-        <height>30</height>
-       </rect>
-      </property>
