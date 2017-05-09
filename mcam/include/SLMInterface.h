#pragma once
#include "Blink_SDK.h"
#include "SLMParents.h"
#include <conio.h>
typedef std::vector<unsigned char>  uchar_vec;

class SLMInterface {
public:
	SLMInterface(Blink_SDK *pSDK) : ramp1_(M * M), ramp2_(M * M), board_number(1)
	{
		pSDK_ = pSDK;
	};

	bool SendTestPhase(unsigned char *parent, int pixel_dimension) 
	{
		uchar_vec ramp1(pixel_dimension * pixel_dimension);
		uchar_vec ramp2(pixel_dimension * pixel_dimension);

		phaseRandom(pixel_dimension, pixel_dimension, ramp1);
		phaseRandom(pixel_dimension, pixel_dimension, ramp2);

		bool result = Precalculate_and_loop(ramp1, ramp2, board_number, *pSDK_);
		return result;
	}

	bool SendPhase(unsigned char *parent)
	{
		unsigned int byte_count = 0U;
		bool okay;
		okay = pSDK_->Write_overdrive_image(board_number, parent);
		okay = okay && pSDK_->Calculate_transient_frames(ramp2_.data(), &byte_count);

		uchar_vec transient1(byte_count);
		okay = okay && pSDK_->Retrieve_transient_frames(transient1.data());

		// Get the SLM into the second phase state, and calculate the transient
		// frames.
		okay = okay && pSDK_->Write_overdrive_image(board_number, ramp2_.data()) &&
			pSDK_->Calculate_transient_frames(ramp1_.data(), &byte_count);

		// Use another std::vector to store the frame sequence.
		uchar_vec transient2(byte_count);
		okay = okay && pSDK_->Retrieve_transient_frames(transient2.data());

		// Now we've completed the pre-calculation, write to the SLM.
		unsigned int i = 0;

		while ((okay) && (!_kbhit()))
		{
			// Switch from second state to first, then back again.
			okay = pSDK_->Write_transient_frames(board_number, transient2.data()) &&
				pSDK_->Write_transient_frames(board_number, transient1.data());

			++i;
			if (!(i % 50))
			{
				printf("Completed cycles: %u\r", i);
			}
		}

		return okay;
	}

private:
	Blink_SDK *pSDK_;
	uchar_vec ramp1_;
	uchar_vec ramp2_;
	bool toggleRamp;
	const int board_number;


	void phaseRandom(const size_t width,
		const size_t height,
		uchar_vec& pixels)
	{
		// This function ASSUMES that pixels.size() is at least width * height.
		unsigned char* pix = pixels.data();

		for (size_t i = 0U; i < height; ++i)    // for each row
		{
			for (size_t j = 0U; j < width; ++j)  // for each column
			{
				*pix++ = static_cast<unsigned char>(static_cast<int> (rand() % 255));
			}
		}

		return;
	}

	// -------------------- Precalculate_and_loop ---------------------------------
	// This function toggles between two ramp images, after pre-calculating the
	// Overdrive frame sequence.
	// ----------------------------------------------------------------------------
	bool Precalculate_and_loop(const uchar_vec& ramp1,
		const uchar_vec& ramp2,
		const int board_number,
		Blink_SDK& sdk)
	{
		puts("\nPrecalculate_and_loop: Press any key to exit.\n");

		// Get the SLM into the first phase state, and calculate the transient
		// frames.
		unsigned int byte_count = 0U;
		bool okay = sdk.Write_overdrive_image(board_number, ramp1.data()) &&
			sdk.Calculate_transient_frames(ramp2.data(), &byte_count);
		// Use a std::vector to store the frame sequence.
		uchar_vec transient1(byte_count);
		okay = okay && sdk.Retrieve_transient_frames(transient1.data());

		// Get the SLM into the second phase state, and calculate the transient
		// frames.
		okay = okay && sdk.Write_overdrive_image(board_number, ramp2.data()) &&
			sdk.Calculate_transient_frames(ramp1.data(), &byte_count);
		// Use another std::vector to store the frame sequence.
		uchar_vec transient2(byte_count);
		okay = okay && sdk.Retrieve_transient_frames(transient2.data());

		// Now we've completed the pre-calculation, write to the SLM.

		unsigned int i = 0;

		while ((okay) && (!_kbhit()))
		{
			// Switch from second state to first, then back again.
			okay = sdk.Write_transient_frames(board_number, transient2.data()) &&
				sdk.Write_transient_frames(board_number, transient1.data());

			++i;
			if (!(i % 50))
			{
				printf("Completed cycles: %u\r", i);
			}
		}

		/*
		if (okay)     // Loop terminated because of a keystroke?
		{
			Consume_keystrokes();
		}
		*/

		return okay;
	}

};

