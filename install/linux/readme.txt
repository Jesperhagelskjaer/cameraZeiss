Zeiss Axiocam USB3.0 Linux driver
---------------------------------

Please copy the file
  axiocam_fx3.rules
to
  /etc/udev/rules.d

then execute

udevadm control --reload-rules

and re-connect any Axiocam USB 3.0 camera

else
==
    reboot
    