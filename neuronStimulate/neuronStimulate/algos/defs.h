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
#define DELAY_MS			4			// Delay in ms laser is turned on while analysing neuron pulses (On)
#define PAUSE_MS			0			// Waiting delay in ms after each iteration (Off)

#define ACTIVE_CHANNEL		31			// Select active channel 0-31
#define FILTER_TYPE			FC_0_1Hz   	// Bandpass filter type low cut fq: 
                                        // BYPASS(0), CUSTOM(1), FC_0_1Hz(2), 
                                        ///FC_1Hz(3), FC_10Hz(4), FC_100Hz(5), FC_300Hz(6)
#define COMMON_REF          2           // Enable common average(1)/median(2) reference filtering
