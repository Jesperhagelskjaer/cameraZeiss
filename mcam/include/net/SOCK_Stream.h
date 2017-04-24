
class SOCK_Stream {

public:
	// Default and copy constructor.
	SOCK_Stream () : handle_(INVALID_SOCKET) {}
	SOCK_Stream (SOCKET h): handle_(h) {}

	// Automatically close the ahandle on destruction
	~SOCK_Stream () 
	{ closesocket (handle_); }

	void close ()
	{ closesocket (handle_); }

	
	// Set/get the underlying SOCKET handle
	void set_handle (SOCKET h) 
	{ handle_ = h; }
	SOCKET get_handle () const 
	{ return handle_; }

	int setBlocking(bool blocking)
	{
		unsigned long mode = blocking ? 0 : 1;
		return ioctlsocket(handle_, FIONBIO, &mode);
	}

	// Regular I/O operations - not implemented
	//size_t recv (void *buf, size_t len, int flags); 
	//size_t send (const char *buf, size_t len, int flags);

	// I/O operation for "short" receives and sends
	size_t recv_n (char *buf, size_t len, int flags)
	{
		size_t n = recv(handle_, buf, len, flags);
		return n;
	}
	size_t send_n (const char *buf, size_t len, int flags)
	{
		size_t n = send(handle_, buf, len, flags);
		return n;
	}

private:
	SOCKET handle_;
};
