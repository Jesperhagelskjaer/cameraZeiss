#pragma once

// If defined then use SLM and LASER
#define SLM_INTERFACE_		1
#define LASER_INTERFACE_    1

// To be defined for testing without Digital Lynx SX and UDP LxRecords
//#define TEST_GENERATOR_		  1

// Used in TemplateImages together with zeiss camera
#define ROWS				50			// Max. hight of image section
#define COLS				50			// Max. width of image section

// Used in SLMParents (h+cpp)
#define M					512			// Matrix size of SLM template
#define MUT_PROPABILITY     exp(-2)	    // Generic mutation propability - exp(-0.52)

// User parameters, set as default in Configuration.h
#define NUM_BINDINGS		4			// M modulus NUM_BINDINGS should be equal to zero !!!!
#define NUM_PARENTS			20			// Number of parents
#define GEN_ITERATIONS		20000		// Number of iterations for genetic algorithm to convergate
#define LASER_PORT			8			// COM port connected to laser
#define LASER_INTENSITY		0.6f		// Intensity of laser when turned on
#define PAUSE_MS			0			// Waiting delay in ms after each iteration

#define DELAY_MS			4			// Delay in ms laser is turned on while analysing neuron pulses
#define ACTIVE_CHANNEL		31			// Select active channel 0-31
#define FILTER_TYPE			FC_0_1Hz   	// Bandpass filter type low cut fq: BYPASS, FC_0_1Hz, FC_1Hz, FC_10Hz, FC_100Hz, FC_300Hz

