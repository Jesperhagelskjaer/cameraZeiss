#ifndef CCAM_H_
#define CCAM_H_

#include <stdint.h>

#ifndef _WIN32
#include <unistd.h>
#include <sys/syscall.h>
#include <limits.h>
#define MAX_PATH PATH_MAX
#endif

#include "mcam_zei.h"
#include "mcam_zei_ex.h"

#endif /* CCAM_H_ */
