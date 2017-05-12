#pragma once
//#include <windows.h>
//#include <stdio.h>
#include <Mmsystem.h>
#pragma comment(lib, "Winmm.lib" )

class CBaseTimer {
public:
    CBaseTimer(int nIntervalMs=1000) { m_nIntervalMs=nIntervalMs; }; // ctor
    ~CBaseTimer()               { Kill(); };                         // dtor
       int m_nID;
       long m_nIntervalMs;

    static void CALLBACK MyTimerProc( UINT uID, UINT uMsg, DWORD dwUser, DWORD dw1, DWORD dw2 ) {
        CBaseTimer* pThis= (CBaseTimer*)dwUser;
        pThis->OnTimer();
    }

    void Start() {  // TBD: Check if 0 (failed)
        m_nID= timeSetEvent( m_nIntervalMs, 10, (LPTIMECALLBACK)MyTimerProc, (DWORD)this, TIME_PERIODIC ); //<<--- Note: this
    }
    void Kill()  {
        ::timeKillEvent( m_nID );
    }
    virtual void OnTimer()=0;  // must supply a fn in derived objects
};
