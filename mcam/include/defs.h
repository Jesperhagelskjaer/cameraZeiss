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

#define RSTART              400			// Number of start modes
#define REND                100         // Number of end modes
#define LAMBDA              1.0			// Constant
//#define MUT_PROPABILITY(n)  ( (RSTART - REND)*exp(-n/LAMBDA) + REND )	    // Generic mutation propability - paper version
#define MUT_PROPABILITY(n)  exp(-2)		// Generic mutation propability - exp(-0.52)

// User parameters, set as default in MCamRemote.cpp
#define NUM_BINDINGS		4			// M modulus NUM_BINDINGS should be equal to zero !!!!
#define NUM_PARENTS			20			// Number of parents
#define GEN_ITERATIONS		1000   		// Number of iterations for genetic algorithm to convergate
#define LASER_PORT			4			// COM port connected to laser
#define LASER_INTENSITY		0.2f		// Intensity of laser when turned on
#define DELAY_MS			150			// Delay in ms laser is turned on before taking image (On)
#define PAUSE_MS			200			// Waiting delay in ms after each iteration before laser turned off (Off)

// Parameters not set in UI
#define NUM_RAND_ITERATIONS 0           // Number of iterations before replacing templates with lowest cost with new random templates (0=turned off)
#define NUM_RAND_TEMPLATES  10			// Number of templates with lowest cost to to be replaced
#define RAND_PROPABILITY    0			// Probability function selecting templates when generating offsprings (1 = logistic probability distribution)

// User parameter for MCAM only
#define LASER_STEP          0.0f		// Step size decreasing laser intensity when pixels are saturated (0=turned off)
#define NUM_SATURATED       0			// Number of pixels (+1) saturated before decreasing laser intensity
#define COST_FUNCTION       0			// 0 - computes cost as sum of pixels in image focus area
									    // 1 - computes cost as sum of pixels divided by mean of local maximum area

// User parameters only used by neuronStimulate 
//#define ACTIVE_CHANNEL		31			// Select active channel 0-31
//#define FILTER_TYPE			FC_0_1Hz   	// Bandpass filter type low cut fq: BYPASS, FC_0_1Hz, FC_1Hz, FC_10Hz, FC_100Hz, FC_300Hz

// Used in MCamRemote only
#define NUM_BETWEEN_SAVE_IMG    20			// Number of iterations before saving image
