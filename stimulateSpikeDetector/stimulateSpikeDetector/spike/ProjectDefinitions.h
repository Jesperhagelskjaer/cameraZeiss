///////////////////////////////////////////////////////////
//  ProjectDefinitions.h
//  Header:          This file defines the setup for this project.
//  Created on:      27-10-2017
//  Original author: MB
///////////////////////////////////////////////////////////
#ifndef PROJECT_DEFINITIONS_H
#define PROJECT_DEFINITIONS_H

/*********************************** GENERAL SETUP ****************************************************/
#define					USE_OPENCV // Use OPENCV for training, when training with C++/OPENCV
//#define                 USE_CUDA_TRAIN // Use CUDA for training

#define					USE_CUDA // The USE_CUDA define must always be enabled for prediction
//#define                 CUDA_VERIFY // ONLY valid with USE_KERNEL_FILTER
#define					PRINT_OUTPUT_INFO

#ifndef USE_CUDA 
#error Only woriking with USE_CUDA defined for single neuron stimulator and prediction
#endif

/*********************************** PATHS ************************************************************/
#define					PATH_TO_CONFIG_FILE					"TestData/projectInfo.bin"
#define					PATH_TO_KILSORT_GT_TRAINING			"TestData/RezInfo.bin"
#define					PATH_TO_TEMPLATES					"TestData/Templates/template_"
#define					PATH_TO_TRAINING_DATA				"TestData/rawData300000x32.bin"
#define					PATH_TO_PREDICTION_DATA				"TestData/rawDataForPrediction300000x32.bin"

/*********************************** SAMPLING *********************************************************/
#define					USED_DATATYPE						float
#define					SAMPLING_FREQUENCY					30000
#define					TRAINING_DATA_TIME					50 // Training time must same size as generated from MATLAB
#define					RUNTIME_DATA_TIME					60 // The runtime/prediction data is assumed to be consecutive to the training data
#define                 RTP_DATA_TIME                       0.005f // Runtime buffer length in seconds, must be equal to DELAY_MS in defs.h!!
#define					TRAINING_DATA_LENGTH				SAMPLING_FREQUENCY*TRAINING_DATA_TIME
#define					RUNTIME_DATA_LENGTH					SAMPLING_FREQUENCY*RUNTIME_DATA_TIME
#define                 RTP_DATA_LENGTH						(SAMPLING_FREQUENCY*RTP_DATA_TIME)
#define					DATA_CHANNELS						32		

/*********************************** MATLAB OUTPUT ****************************************************/
#define					KILOSORT_ST3_WIDTH_USED				2
#define					CONFIG_FILE_LENGTH					1

/*********************************** TEMPLATES ********************************************************/
#define					MAXIMUM_NUMBER_OF_TEMPLATES			64
#define					TEMPLATE_ORIGINAL_LENGTH			61
#define					TEMPLATE_ORIGINAL_WIDTH				DATA_CHANNELS
#define					TEMPLATE_CROPPED_LENGTH				17
#define					TEMPLATE_CROPPED_WIDTH				9

/*********************************** DRIFT HANDLING ***************************************************/
#define					NUMBER_OF_DRIFT_CHANNELS_HANDLED	0 // only 1 has been tested! OpenCV does not support drift currently
#ifdef USE_OPENCV 
#if NUMBER_OF_DRIFT_CHANNELS_HANDLED > 0
#error THIS IS NOT ALLOWED!
#endif
#endif

/*********************************** CHANNEL FILTERING ************************************************/
#define					NUMBER_OF_A_COEFF					6
#define					NUMBER_OF_B_COEFF					4

/*********************************** KERNEL FILTERING *************************************************/
//#define                 USE_KERNEL_FILTER
#define					DEFAULT_KERNEL_DIM					3 // Equals 3x3 - Currently only supported!

/*********************************** TRAINED THRESHOLD ************************************************/
#define					PRECISION_WEIGHT					0.7f
#define					RECALL_WEIGHT						0.3f
#define					ACCEPTED_TIMELINE_SLACK				3
#define					NUMBER_OF_THRESHOLDS_TO_TEST		40
#define				    MINIMUM_THRESHOLD_TO_TEST			0.2f
#define				    MAXIMUM_THRESHOLD_TO_TEST			1
#define					MAXIMUM_TRAINING_SAMPLES			TRAINING_DATA_TIME*1000 // inidcats a Maximum 1000 spikes per template per second
#define					MAXIMUM_PREDICTION_SAMPLES			(RTP_DATA_TIME*4000) // inidcats a Maximum peak 4000 spikes per template per second
//#define					MAXIMUM_PREDICTION_SAMPLES			RUNTIME_DATA_TIME*1000 // inidcats an average of 1000 spikes per template per second
//#define					MAXIMUM_PREDICTION_SAMPLES			TRAINING_DATA_TIME*1000 // inidcats a Maximum 1000 spikes per template per second

/*********************************** CUDA OPTIMIZATION ************************************************/
#define					MAXIMUM_NUMBER_OF_THREADS						1024
#define					MAXIMUM_NUMBER_OF_THREADS_COMPARING				500
#define					MAXIMUM_NUMBER_OF_THREADS_DRIFT_HANDLING		1024

#endif