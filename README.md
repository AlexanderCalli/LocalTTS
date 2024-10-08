Local Text-to-Speech (TTS) System
==================================

This project implements a local text-to-speech system using a pre-trained model. It allows users to generate speech from text input using different voice models.

Features:
---------
- Initialize a local TTS client
- Load different voice models
- Generate speech from text input
- Support for multiple languages (default: English)
- Customizable voice selection

Main Components:
----------------

1. TTS Client Initialization:
   The script initializes a TTS client, connecting to a local server running on port 5003.

2. Model Loading:
   The 'load_model' function loads a specific voice model using the provided checkpoint, config, vocab, and speaker files.

3. Speech Generation:
   The 'generate_speech' function takes text input and generates audio output using the loaded model. It supports various parameters for customizing the generation process.

4. Voice Selection:
   The 'get_voice_path' function allows users to select from available voice models interactively.

Usage:
------
1. The script initializes the TTS client.
2. Users select a voice model from the available options.
3. The chosen model is loaded.
4. Text input is read from an 'input.txt' file in the same directory.
5. The system generates speech based on the input text.
6. The generated audio is saved as a temporary WAV file in the project directory.

Output:
-------
The script generates a temporary WAV file containing the synthesized speech and prints the path to this file along with the reference audio path.

Note:
-----
This system requires a local TTS server running on port 5003. Ensure that the server is set up and running before using this script.

Code Overview:
--------------

def init_tts_client():
    # Initializes the TTS client

def load_model(client, model_path, config_path, vocab_path, speakers_path):
    # Loads a specific voice model

def generate_speech(client, text, language="en", reference_audio=None):
    # Generates speech from input text

def get_voice_path():
    # Allows users to select a voice model

if __name__ == "__main__":
    # Main execution flow:
    # 1. Initialize client
    # 2. Get voice path
    # 3. Load model
    # 4. Read input text
    # 5. Generate speech
    # 6. Save output to temporary file

The script reads text from 'input.txt', processes it through the selected TTS model, and outputs a WAV file with the generated speech.