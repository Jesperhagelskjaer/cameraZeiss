
class INET_Addr {

public:
	INET_Addr () 
	{ memset(&addr_, 0, sizeof addr_); };

	INET_Addr (u_short port, u_long addr) 
    {
		// Set up the address to become a server
		memset(&addr_, 0, sizeof addr_);
		addr_.sin_family = AF_INET;
		addr_.sin_port = htons (port);
		addr_.sin_addr.s_addr = htonl (addr);
	}
	
	INET_Addr (u_short port) 
    {
		struct addrinfo hints, *result;
		char strPort[20];
		sprintf_s(strPort, "%d", port);

		// Set up the address to become a server
		ZeroMemory(&hints, sizeof (hints));
		hints.ai_family = AF_INET;
		hints.ai_socktype = SOCK_STREAM;
		hints.ai_protocol = IPPROTO_TCP;
		hints.ai_flags = AI_PASSIVE;

		// Resolve the local address and port to be used by the server
		int iResult = getaddrinfo(NULL, (PCSTR)port, &hints, &result);
		if (iResult != 0) {
			printf("getaddrinfo failed: %d\n", iResult);
		}

		memset(&addr_, 0, sizeof addr_);
		//addr_.sin_addr = result->ai_addr; // To be fixed!!!
		addr_.sin_family = result->ai_family;
		addr_.sin_port = port;
	}

	u_short get_port() const 
	{ return addr_.sin_port; }

	u_long get_ip_addr () const
	{ return addr_.sin_addr.s_addr; }

	sockaddr *addr ()
	{ return reinterpret_cast <sockaddr *> (&addr_); }

	size_t size () const
	{ return sizeof (addr_); }

private:
	sockaddr_in addr_;
};

