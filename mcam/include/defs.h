#pragma once

// Used in TemplateImages
#define ROWS			50	// Max. hight of image section
#define COLS			50	// Max. width of image section

// Used in SLMParents
#define NUM_PARENTS		20	 // Number of parents
#define NUM_ITERATIONS  60	 // Number of iterations
#define BIND			4    // M modulus BIND should be equal to zero !!!!
#define M				512  // Matrix size of SLM timeplate

// Used in MCamRemote
#define NUM_BETWEEN_SAVE_IMG   20 // Number of iterations before saving image

// If defined then use SLM
#define SLM_INTERFACE_		1
//#define TEST_WITHOUT_CAMERA_	1
