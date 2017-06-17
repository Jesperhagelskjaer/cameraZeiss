#pragma once

// Used in TemplateImages
#define ROWS			50	// Max. hight of image section
#define COLS			50	// Max. width of image section

// Used in SLMParents
#define NUM_PARENTS		20	 // Number of parents
#define NUM_ITERATIONS  1000	 // Number of iterations
#define BIND			4    // M modulus BIND should be equal to zero !!!!
#define M				512  // Matrix size of SLM timeplate

// If defined then use SLM
#define SLM_INTERFACE_		1
//#define TEST_WITHOUT_CAMERA_	1
#define LASER_INTERFACE_    1
#define LASER_PORT          8
#define LASER_INTENSITY     0.2f    // Intensity of laser when turned on

// Used in MCamRemote
#define NUM_BETWEEN_SAVE_IMG   20 // Number of iterations before saving image
