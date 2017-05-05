/****************************************************************************
** Meta object code from reading C++ file 'MCamImage.hpp'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../include/MCamImage.hpp"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MCamImage.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MCamImage[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: signature, parameters, type, tag, flags
      19,   11,   10,   10, 0x05,
      53,   47,   10,   10, 0x05,
      89,   77,   73,   10, 0x05,

       0        // eod
};

static const char qt_meta_stringdata_MCamImage[] = {
    "MCamImage\0\0rateStr\0updateTransferRate(QString)\0"
    "start\0contShotStart(bool)\0int\0cameraIndex\0"
    "cameraSelected(long)\0"
};

void MCamImage::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        MCamImage *_t = static_cast<MCamImage *>(_o);
        switch (_id) {
        case 0: _t->updateTransferRate((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 1: _t->contShotStart((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 2: { int _r = _t->cameraSelected((*reinterpret_cast< long(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        default: ;
        }
    }
}

const QMetaObjectExtraData MCamImage::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject MCamImage::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_MCamImage,
      qt_meta_data_MCamImage, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &MCamImage::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *MCamImage::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *MCamImage::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MCamImage))
        return static_cast<void*>(const_cast< MCamImage*>(this));
    return QObject::qt_metacast(_clname);
}

int MCamImage::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    }
    return _id;
}

// SIGNAL 0
void MCamImage::updateTransferRate(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void MCamImage::contShotStart(bool _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
int MCamImage::cameraSelected(long _t1)
{
    int _t0;
    void *_a[] = { const_cast<void*>(reinterpret_cast<const void*>(&_t0)), const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
    return _t0;
}
QT_END_MOC_NAMESPACE
