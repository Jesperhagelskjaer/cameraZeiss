/**
 * @file mcamloger.h
 * @author ggraf
 * @date 17.02.2015
 *
 * @brief header file for mcam logger implementation
 *
 * Copyright CCD Videometrie GmbH 2015, All rights reserved.
 */

#define MCAM_EXT_LOGGER_INIT
#define MCAM_EXT_LOGGER_DEINIT

#define NOCAMI 0xfffe

// logger FILE* or empty
#define MCAM_LOGP

// logger
#define MCAM_LOGGER

#define MCAM_DEBUG_ERROR       0x00000001
#define MCAM_DEBUG_WARN        0x00000002
#define MCAM_DEBUG_INFO        0x00000004
#define MCAM_DEBUG_DEBUG       0x00000008
#define MCAM_DEBUG_STATUS      0x00000010

extern "C" {
int mcamLoggingInit(int append, const char *pfilename, long debugMaxFileSize);
int mcamLoggingDeInit();
int mcamSetLogLevel(uint32_t parmAxcamLevel);

int mcam_log_error(const char *log_fkt_name, unsigned short cameraindex, const char* message, ...);
int mcam_log_warn(const char *log_fkt_name, unsigned short cameraindex, const char* message, ...);
int mcam_log_info(const char *log_fkt_name, unsigned short cameraindex, const char* message, ...);
int mcam_log_debug(const char *log_fkt_name, unsigned short cameraindex, const char* message, ...);
int mcam_log_status(const char *log_fkt_name, unsigned short cameraindex, const char* message, ...);

}
;

#define MCAM_LOGF_ERROR(MESSAGE, ...) mcam_log_error(log_fkt_name, NOCAMI, MESSAGE, ##__VA_ARGS__)
#define MCAM_LOGF_WARN(MESSAGE, ...) mcam_log_warn(log_fkt_name, NOCAMI, MESSAGE, ##__VA_ARGS__)
#define MCAM_LOGF_INFO(MESSAGE, ...) mcam_log_info(log_fkt_name, NOCAMI, MESSAGE, ##__VA_ARGS__)
#define MCAM_LOGF_DEBUG(MESSAGE, ...) mcam_log_debug(log_fkt_name, NOCAMI, MESSAGE, ##__VA_ARGS__)
#define MCAM_LOGF_STATUS(MESSAGE, ...) mcam_log_status(log_fkt_name, NOCAMI, MESSAGE, ##__VA_ARGS__)

#define MCAM_LOGFI_ERROR(MESSAGE, ...) mcam_log_error(log_fkt_name, (unsigned short) cameraIndex, MESSAGE, ##__VA_ARGS__)
#define MCAM_LOGFI_WARN(MESSAGE, ...) mcam_log_warn(log_fkt_name, (unsigned short) cameraIndex, MESSAGE, ##__VA_ARGS__)
#define MCAM_LOGFI_INFO(MESSAGE, ...) mcam_log_info(log_fkt_name, (unsigned short) cameraIndex, MESSAGE, ##__VA_ARGS__)
#define MCAM_LOGFI_DEBUG(MESSAGE, ...) mcam_log_debug(log_fkt_name, (unsigned short) cameraIndex, MESSAGE, ##__VA_ARGS__)
#define MCAM_LOGFI_STATUS(MESSAGE, ...) mcam_log_status(log_fkt_name, (unsigned short) cameraIndex, MESSAGE, ##__VA_ARGS__)

// generic level logging (see alternative used below)
#define MCAM_LOG_LEVEL(LEVEL, MCAM_LOGGER) if ((mcam_debug_level & (_mcam_level=LEVEL)) != 0) MCAM_LOGGER

#define MCAM_LOGPS "%s"
#define MCAM_LOG_INIT(NAME) const char *log_fkt_name = NAME;
#define MCAM_LOGPP ""

// log variant with cameraindex!
#define MCAM_LOGPSI "CAM#%d "
#define MCAM_LOGPPI cameraindex

#define MCAM_LOG_CHECK(LEVEL) ((mcam_debug_level & LEVEL) != 0)
