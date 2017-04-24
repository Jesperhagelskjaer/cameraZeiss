class SOCK_Connector {

public:
	SOCK_Connector () : handle_(INVALID_SOCKET) {}

	// Initialize a passive-mode acceptor socket
	SOCK_Connector (INET_Addr &addr) 
	{
		open(addr);	
	}

	// A second method to initialize a passive-mode
	// acceptor socket, analogously to the constructor
	void open (INET_Addr &sock_addr) 
	{	
		sock_addr_ = sock_addr;
		// Create a local endpoint of communication
		handle_ = socket (PF_INET, SOCK_STREAM, 0);
		// Associate address with endpoint
		bind (handle_, sock_addr.addr(), sock_addr.size());
	}

	void connecting (SOCK_Stream &s)
	{
		// Connect to server.
		int iResult = connect( handle_, sock_addr_.addr(), sock_addr_.size());
		if (iResult == SOCKET_ERROR) {
			closesocket(handle_);
			handle_ = INVALID_SOCKET;
		}
		s.set_handle(handle_);

		// Should really try the next address returned by getaddrinfo
		// if the connect call failed
		// But for this simple example we just free the resources
		// returned by getaddrinfo and print an error message
		//freeaddrinfo(result);
	}

	void close()
	{
		if (handle_ != INVALID_SOCKET) {
			closesocket(handle_);
			handle_ = INVALID_SOCKET;
		}
	}


private:
	SOCKET handle_; // Socket handle factory
	INET_Addr sock_addr_; // Address
};
