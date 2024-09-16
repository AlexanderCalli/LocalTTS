import os
import tempfile
from gradio_client import Client

def init_tts_client():
    return Client("http://localhost:5003/")

def load_model(client, checkpoint, config, vocab, speaker):
    client.predict(
        checkpoint,
        config,
        vocab,
        speaker,
        api_name="/load_model"
    )

def generate_speech(client, text, language="en", reference_audio=None):
    result = client.predict(
        language,
        text,
        reference_audio,
        0.7,  # temperature
        0,    # length_penalty
        1,    # repetition_penalty
        50,   # top_k
        0.9,  # top_p
        True, # Enable text splitting
        True, # Use inference settings from config
        api_name="/run_tts"
    )
    return result

def get_voice_path():
    base_path = r"C:\Programming\xtts-finetune-webui\finetune_models"
    voices = {
        "Ready": os.path.join(base_path, "ready"),
        "Jessica": os.path.join(base_path, "jessica", "ready"),
        "Male": os.path.join(base_path, "male", "ready")
    }
    
    print("Available voices:")
    for i, voice in enumerate(voices.keys(), 1):
        print(f"{i}. {voice}")
    
    while True:
        choice = input("Enter the number of the voice you want to use: ")
        try:
            choice = int(choice)
            if 1 <= choice <= len(voices):
                selected_voice = list(voices.keys())[choice - 1]
                return voices[selected_voice]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def read_input_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

if __name__ == "__main__":
    client = init_tts_client()
    
    voice_path = get_voice_path()
    
    # Load the model with paths for the selected voice
    load_model(
        client,
        os.path.join(voice_path, "model.pth"),
        os.path.join(voice_path, "config.json"),
        os.path.join(voice_path, "vocab.json"),
        os.path.join(voice_path, "speakers_xtts.pth")
    )
    
    # Read text from input.txt
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    text_to_speak = read_input_file(input_file_path)
    
    # Generate speech
    result = generate_speech(
        client,
        text_to_speak,
        language="en",
        reference_audio=os.path.join(voice_path, "reference.wav")
    )
    
    # Create a temporary file in the project directory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=os.path.dirname(__file__)) as temp_file:
        temp_file_path = temp_file.name
    
    # Move the generated audio to the temporary file
    os.replace(result[1], temp_file_path)
    
    print("Generated audio saved to:", temp_file_path)
    print("Reference audio path:", result[2])