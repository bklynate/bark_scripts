import os
import threading
from datetime import datetime
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import spacy
from pydub import AudioSegment

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Download and load all models from Bark
preload_models()


def create_output_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'./output_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def process_text_snippet(text_snippet, snippet_index, output_dir):
    try:
        audio_array = generate_audio(
            text_snippet, history_prompt="v2/en_speaker_9")
        filename = f'bark_generation_{snippet_index}.wav'
        full_path = os.path.join(output_dir, filename)
        print(f'Saving to {full_path}')
        write_wav(full_path, SAMPLE_RATE, audio_array)
        print(f'Audio snippet {snippet_index} saved as {filename}')
    except Exception as e:
        with open('error_log.txt', 'a') as f:
            f.write(f'Error processing snippet {snippet_index}: {e}\n')


def chunk_text(text):
    doc = nlp(text)
    chunks = [sent.text for sent in doc.sents]
    return chunks


def trim_silence(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    trimmed_audio = audio.strip_silence()
    trimmed_audio.export(audio_path, format="wav")


def add_overlap(audio_path1, audio_path2, overlap_duration):
    audio1 = AudioSegment.from_wav(audio_path1)
    audio2 = AudioSegment.from_wav(audio_path2)
    # Ensure the position is non-negative
    position = max(len(audio1) - overlap_duration, 0)
    overlapped_audio = audio1.overlay(audio2, position=position)
    overlapped_audio.export(
        f'{audio_path1}_overlapped.wav', format="wav")  # Modified this line


def merge_audio_files(output_dir):
    audio_files = [os.path.join(output_dir, f)
                   for f in os.listdir(output_dir) if f.endswith('.wav')]
    # Filter out overlapped files
    audio_files = [f for f in audio_files if 'overlapped' not in f]
    # Now sort the remaining files
    audio_files = sorted(audio_files, key=lambda x: int(
        x.split('_')[-1].split('.')[0].replace('bark_generation_', '')))
    combined_audio = AudioSegment.empty()
    for audio_file in audio_files:
        audio_segment = AudioSegment.from_wav(audio_file)
        combined_audio += audio_segment
        # Debug print statement
        print(f'Processing {audio_file}, duration: {len(audio_segment)} ms')
    combined_audio.export(os.path.join(
        output_dir, 'combined_audio.wav'), format='wav')
    # Debug print statement
    print(f'Combined audio duration: {len(combined_audio)} ms')


def main():
    # Your text_prompt here...
    text_prompt = """
    "Hey there, lovely people! Welcome back to our cozy little finance corner on the web. Today, we're unraveling the financial threads of Palantir Technologies. Is it a sparkling BUY, a steady hold, or a hard pass? If you’re as excited as we are for some stock sleuthing, hit that subscribe button for more investment detective work. Now, let’s dive in!"

    "Palantir has been buzzing on the radar for its tech wizardry, but does its financial health match the hype? Over the past quartet of years, it’s been climbing the revenue ladder, boasting a 47% hike from 2019 to 2020. Yet, the trail wasn’t all roses."

    "The cash flow tide was in the red for 2019 and 2020, but 2021 brought a change in currents. [clears throat] Our friend, the Free Cash Flow Yield, swam from the negatives to the positives, although it’s still paddling below that dreamy 7% mark."
    """
    # Split text into manageable chunks
    text_chunks = chunk_text(text_prompt)

    # Create a new output directory for this run
    output_dir = create_output_directory()

    # Create threads to process text chunks
    threads = []
    for i, text_chunk in enumerate(text_chunks):
        print(f'Starting thread for chunk {i}')  # Print when starting a thread
        thread = threading.Thread(
            target=process_text_snippet, args=(text_chunk, i, output_dir))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # After generating all audio snippets
    for i in range(len(text_chunks) - 1):
        audio_path1 = os.path.join(output_dir, f'bark_generation_{i}.wav')
        audio_path2 = os.path.join(output_dir, f'bark_generation_{i + 1}.wav')
        trim_silence(audio_path1)
        add_overlap(audio_path1, audio_path2, 200)  # 200 ms overlap

    # Don't forget to trim silence from the last audio snippet
    trim_silence(os.path.join(
        output_dir, f'bark_generation_{len(text_chunks) - 1}.wav'))

    # Merge all audio snippets into one continuous audio file
    merge_audio_files(output_dir)

    print('All audio snippets processed and merged.')


if __name__ == "__main__":
    main()
