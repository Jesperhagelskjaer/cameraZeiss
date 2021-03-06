// SLMInterface.h
// Kim Bjerge
#pragma once
#include "Blink_SDK.h"
#include "SLMParents.h"
#include <conio.h>
#define SLM_INTERFACE_ 1

typedef std::vector<unsigned char>  uchar_vec;

class SLMInterface {
public:

	SLMInterface(Blink_SDK *pSDK) :
		oldPhase(M * M),
		board_number(1)
	{
		pSDK_ = pSDK;
		ResetInterface();
	}

	SLMInterface() :
		oldPhase(M * M),
		board_number(1)
	{
		const unsigned int bits_per_pixel = 8U;
		const unsigned int pixel_dimension = 512U;
		const bool         is_nematic_type = true;
		const bool         RAM_write_enable = true;
		const bool         use_GPU_if_available = true;
		const char* const  regional_lut_file = "C:/Program Files/Meadowlark Optics/Blink OverDrive Plus/LUT Files/slm4037_at635_regional.txt"; //"SLM_regional_lut.txt"; 
		
		pSDK_ = new Blink_SDK(bits_per_pixel, pixel_dimension, &n_boards_found_,
			                  &constructed_okay_, is_nematic_type, RAM_write_enable,
			                  use_GPU_if_available, 20U, regional_lut_file);
		ResetInterface();
	}

	~SLMInterface()
	{
		pSDK_->SLM_power(false);
	}

	bool ResetInterface(void) 
	{
		bool okay = false;

		memset(oldPhase.data(), 0, M*M);

		if (pSDK_->Is_slm_transient_constructed()) {
			enum { e_n_true_frames = 5 };
			pSDK_->Set_true_frames(e_n_true_frames);
			pSDK_->SLM_power(true);
			okay = pSDK_->Load_linear_LUT(board_number);
		}
		return okay;
	}

	bool SendTestPhase(unsigned char *parent, int pixel_dimension) 
	{
		uchar_vec ramp1(pixel_dimension * pixel_dimension);
		uchar_vec ramp2(pixel_dimension * pixel_dimension);

		phaseRandom(pixel_dimension, pixel_dimension, ramp1);
		phaseRandom(pixel_dimension, pixel_dimension, ramp2);

		bool result = Precalculate_and_loop(ramp1, ramp2, board_number, *pSDK_);
		return result;
	}

	bool SendPhase(unsigned char *phase)
	{
		unsigned int byte_count = 0U;
		bool okay = true;

		//okay = pSDK_->Write_overdrive_image(board_number, oldPhase.data());
		okay = okay && pSDK_->Calculate_transient_frames(phase, &byte_count);
		uchar_vec transient(byte_count);

		okay = okay && pSDK_->Retrieve_transient_frames(transient.data());
		okay = okay && pSDK_->Write_transient_frames(board_number, transient.data());
		std::copy(phase, phase + (M*M), oldPhase.begin());

		return okay;
	}

private:
	uchar_vec oldPhase;

	// SLM SDK variables
	Blink_SDK *pSDK_;
	const int board_number;
	unsigned int n_boards_found_;
	bool         constructed_okay_;

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

	void Consume_keystrokes()
	{
		// Get and throw away the character(s) entered on the console.
		int k = 0;
		while ((!k) || (k == 0xE0))  // Handles arrow and function keys.
		{
			k = _getch();
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

		if (okay)     // Loop terminated because of a keystroke?
		{
			Consume_keystrokes();
		}

		return okay;
	}

};

