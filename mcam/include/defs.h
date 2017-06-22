#pragma once

// If defined then use SLM and LASER
#define SLM_INTERFACE_		1
#define LASER_INTERFACE_    1

// To be defined for testing without Zeiss Camera
//#define TEST_WITHOUT_CAMERA_  1

// Used in TemplateImages together with zeiss camera
#define ROWS				50			// Max. hight of image section
#define COLS				50			// Max. width of image section

// Used in SLMParents (h+cpp)
#define M					512			// Matrix size of SLM template
#define MUT_PROPABILITY     exp(-2)	    // Generic mutation propability - exp(-0.52)

// User parameters, set as default in MCamRemote.cpp
#define NUM_BINDINGS		4			// M modulus NUM_BINDINGS should be equal to zero !!!!
#define NUM_PARENTS			20			// Number of parents
#define GEN_ITERATIONS		1000   		// Number of iterations for genetic algorithm to convergate
#define LASER_PORT			8			// COM port connected to laser
#define LASER_INTENSITY		0.6f		// Intensity of laser when turned on
#define DELAY_MS			4			// Delay in ms laser is turned on before taking image
#define PAUSE_MS			0			// Waiting delay in ms after each iteration before laser turned off

// User parameters only used by neuronStimulate 
//#define ACTIVE_CHANNEL		31			// Select active channel 0-31
//#define FILTER_TYPE			FC_0_1Hz   	// Bandpass filter type low cut fq: BYPASS, FC_0_1Hz, FC_1Hz, FC_10Hz, FC_100Hz, FC_300Hz

// Used in MCamRemote only
#define NUM_BETWEEN_SAVE_IMG    20			// Number of iterations before saving image
