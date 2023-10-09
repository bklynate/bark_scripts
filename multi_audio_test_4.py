import threading
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import spacy
from pydub import AudioSegment

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Download and load all models from Bark
preload_models()

def process_text_snippet(text_snippet, snippet_index):
    try:
        audio_array = generate_audio(text_snippet, history_prompt="v2/en_speaker_9")
        filename = f'bark_generation_{snippet_index}.wav'
        full_path = f'./{filename}'
        print(f'Saving to {full_path}')
        write_wav(full_path, SAMPLE_RATE, audio_array)
        print(f'Audio snippet {snippet_index} saved as {filename}')
    except Exception as e:
        print(f'Error processing snippet {snippet_index}: {e}')

def chunk_text(text):
    doc = nlp(text)
    chunks = [sent.text for sent in doc.sents]
    return chunks

def trim_silence(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    # Placeholder - implement your own silence trimming function
    trimmed_audio = audio
    trimmed_audio.export(audio_path, format="wav")

def add_overlap(audio_path1, audio_path2, overlap_duration):
    audio1 = AudioSegment.from_wav(audio_path1)
    audio2 = AudioSegment.from_wav(audio_path2)
    overlapped_audio = audio1.overlay(audio2, position=-overlap_duration)
    overlapped_audio.export(audio_path1, format="wav")

# Your text_prompt here...
text_prompt = """
"Hey everyone, welcome back to our channel. Today we're looking at Palantir Technologies' financials to see if it's a BUY, hold, or a pass. Don’t forget to hit subscribe for more stock analysis and tips. [music] Let's jump right in."

"Palantir has caught the attention of many due to its innovative tech approach. But what about its financial health? Over the last four years, there's been a steady revenue growth, with a whopping 47% increase from 2019 to 2020. However, it wasn’t all positive.

The company saw negative cash flows in 2019 and 2020, but things took a turn in 2021. [clears throat] The Free Cash Flow Yield moved from negative to positive, although it's still below the ideal 7% mark.

Now, onto the margins. The Net Margin improved from a concerning -106.75% in 2020 to -19.61% in 2022. The Earnings Per Share is still negative at -0.02, showing the company isn’t generating positive earnings just yet.

Looking at the balance sheet, the Debt to Equity Ratio sits at a comfortable 0.319, which is a lower level of debt and that’s good. [clears throat] However, the Price to Book Ratio at 11.64 shows the stock is trading at a PREMIUM compared to its book value.

Both the Return on Equity and Return on Assets are in the negative, indicating some inefficiencies in generating income. [sighs]

Market sentiment is mixed. Some analysts are advising to hold the stock, with a few recent reports having a bearish outlook. The target prices suggested are below the current trading range, which is something to consider.

Despite these financial challenges, Palantir's innovation could potentially give it an edge in the competitive tech sector.

So, is Palantir a good investment? It has strengths in revenue growth and innovation, but financial fundamentals and market sentiment suggest a cautious approach. It’s important to weigh the growth prospects against the financial realities before deciding.

That’s all for today’s analysis. Don’t forget to like, share, and subscribe for more insights. [music] Until next time, happy investing!"

"Stay tuned for more stock analysis and remember, invest wisely! [music]"
"""

# Split text into manageable chunks
text_chunks = chunk_text(text_prompt)

# Create threads to process text chunks
threads = []
for i, text_chunk in enumerate(text_chunks):
    print(f'Starting thread for chunk {i}')  # Print when starting a thread
    thread = threading.Thread(
        target=process_text_snippet, args=(text_chunk, i))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# After generating all audio snippets
for i in range(len(text_chunks) - 1):
    trim_silence(f'./bark_generation_{i}.wav')
    add_overlap(f'./bark_generation_{i}.wav', f'./bark_generation_{i + 1}.wav', 200)  # 200 ms overlap

# Don't forget to trim silence from the last audio snippet
trim_silence(f'./bark_generation_{len(text_chunks) - 1}.wav')

print('All audio snippets processed.')
