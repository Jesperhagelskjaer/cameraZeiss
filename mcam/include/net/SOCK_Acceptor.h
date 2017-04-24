
class SOCK_Acceptor {

public:
	SOCK_Acceptor () : handle_(INVALID_SOCKET) {}
	
	// Initialize a passive-mode acceptor socket
	SOCK_Acceptor (INET_Addr &addr) 
	{
		open(addr);
	}

	// A second method to initialize a passive-mode
	// acceptor socket, analogously to the constructor
	void open (INET_Addr &sock_addr) 
	{	
		// Create a local endpoint of communication
		handle_ = socket (PF_INET, SOCK_STREAM, 0);
		// Associate address with endpoint
		bind (handle_, sock_addr.addr(), sock_addr.size());
		// Make endpoint listen for connections
		listen(handle_, 5);
	}

	// Accept a connection and initialize the <stream>
	void accepting (SOCK_Stream &s) 
	{
		s.set_handle(accept(handle_, 0, 0));
    }
private:
	SOCKET handle_; // Socket handle factory
};
