#pragma once
#include <cstdint>

// Defines for Digital Lynx SX system, number of input boards and channels
#define NUM_BOARDS      1  // Number of input boards in Digital Lynx SX
#define NUM_CHANNELS   32  // Number of channels in input board
#define RESERVED_SIZE  10  // Reserved space in Lynx Record header

typedef struct LHEADER
{
	uint32_t start;
	uint32_t packetId;
	uint32_t size;
	uint32_t timestampHigh;
	uint32_t timestampLow;
	uint32_t systemStatus;
	uint32_t ttlIO;
	uint32_t reserved[RESERVED_SIZE];
} LxHeader;

typedef struct LBOARD_DATA
{
	int32_t data[NUM_CHANNELS];
} LxBoardData;

typedef struct LBOARDS_DATA
{
	LxBoardData board[NUM_BOARDS];
} LxBoardsData;

typedef struct LRECORD
{
	LxHeader header;
	LxBoardData board[NUM_BOARDS];
	uint32_t checksum;
} LxRecord;

