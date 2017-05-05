/****************************************************************************
** Meta object code from reading C++ file 'MCamCameraIF.hpp'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../include/MCamCameraIF.hpp"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MCamCameraIF.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MCamCameraIF[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       5,       // signalCount

 // signals: signature, parameters, type, tag, flags
      22,   14,   13,   13, 0x05,
      56,   44,   40,   13, 0x05,
      75,   13,   13,   13, 0x05,
      91,   13,   40,   13, 0x05,
     115,   13,   40,   13, 0x05,

       0        // eod
};

static const char qt_meta_stringdata_MCamCameraIF[] = {
    "MCamCameraIF\0\0enabled\0setBusyLock(bool)\0"
    "int\0cameraIndex\0selectCamera(long)\0"
    "updateDevices()\0showMessageBox(QString)\0"
    "dismissMessageBox()\0"
};

void MCamCameraIF::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        MCamCameraIF *_t = static_cast<MCamCameraIF *>(_o);
        switch (_id) {
        case 0: _t->setBusyLock((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: { int _r = _t->selectCamera((*reinterpret_cast< long(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 2: _t->updateDevices(); break;
        case 3: { int _r = _t->showMessageBox((*reinterpret_cast< QString(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 4: { int _r = _t->dismissMessageBox();
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        default: ;
        }
    }
}

const QMetaObjectExtraData MCamCameraIF::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject MCamCameraIF::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_MCamCameraIF,
      qt_meta_data_MCamCameraIF, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &MCamCameraIF::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *MCamCameraIF::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *MCamCameraIF::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MCamCameraIF))
        return static_cast<void*>(const_cast< MCamCameraIF*>(this));
    return QObject::qt_metacast(_clname);
}

int MCamCameraIF::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void MCamCameraIF::setBusyLock(bool _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
int MCamCameraIF::selectCamera(long _t1)
{
    int _t0;
    void *_a[] = { const_cast<void*>(reinterpret_cast<const void*>(&_t0)), const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
    return _t0;
}

// SIGNAL 2
void MCamCameraIF::updateDevices()
{
    QMetaObject::activate(this, &staticMetaObject, 2, 0);
}

// SIGNAL 3
int MCamCameraIF::showMessageBox(QString _t1)
{
    int _t0;
    void *_a[] = { const_cast<void*>(reinterpret_cast<const void*>(&_t0)), const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
    return _t0;
}

// SIGNAL 4
int MCamCameraIF::dismissMessageBox()
{
    int _t0;
    void *_a[] = { const_cast<void*>(reinterpret_cast<const void*>(&_t0)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
    return _t0;
}
QT_END_MOC_NAMESPACE
