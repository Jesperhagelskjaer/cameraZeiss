#pragma once

// Used in TemplateImages
#define ROWS			50	// Max. hight of image section
#define COLS			50	// Max. width of image section

// Used in SLMParents
#define NUM_PARENTS		20    // Number of parents
#define NUM_ITERATIONS  20000 // Number of iterations, not used (for camera test)
#define BIND			4     // M modulus BIND should be equal to zero !!!!
#define M				512   // Matrix size of SLM timeplate

// If defined then use SLM
//#define SLM_INTERFACE_		1
#define LASER_INTEFACE_         1

// To be defined for testing without Digital Lynx SX and UDP LxRecords
#define TEST_GENERATOR_			1

// User parameters 
#define GEN_ITERATIONS	20000	// Number of iterations for genetic algorithm to convergate
#define LASER_PORT		8		// COM port connected to laser
#define DELAY_MS		4		// Delay in ms to turn laser on
#define ACTIVE_CHANNEL  31		// Select channel 0-31

