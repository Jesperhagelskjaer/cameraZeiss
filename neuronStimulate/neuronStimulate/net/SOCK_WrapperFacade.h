#ifndef SOCK_WRAPPERFACADE_H
#define SOCK_WRAPPERFACADE_H

#include <winsock2.h>
#include <ws2tcpip.h>
#include <Windows.h>
#include <system_error>
#include <string>
#include <iostream>

#pragma comment(lib, "Ws2_32.lib")

#include "INET_Addr.h"
#include "SOCK_Stream.h"
#include "SOCK_Acceptor.h"
#include "SOCK_Connector.h"
#include "SOCK_UDP.h"

class WSASession
{
public:
    WSASession()
    {
        int ret = WSAStartup(MAKEWORD(2, 2), &data);
        if (ret != 0)
            throw std::system_error(WSAGetLastError(), std::system_category(), "WSAStartup Failed");
    }
    ~WSASession()
    {
        WSACleanup();
    }
private:
    WSAData data;
};

#endif
