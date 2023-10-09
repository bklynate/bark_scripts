import threading
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Download and load all models from Bark
preload_models()

def process_text_snippet(text_snippet, snippet_index):
    try:
        audio_array = generate_audio(text_snippet)  # No await keyword here
        filename = f'bark_generation_{snippet_index}.wav'
        full_path = f'./{filename}'  # Adjust this path as necessary
        print(f'Saving to {full_path}')  # Print the complete file path
        write_wav(full_path, SAMPLE_RATE, audio_array)
        print(f'Audio snippet {snippet_index} saved as {filename}')
    except Exception as e:
        print(f'Error processing snippet {snippet_index}: {e}')

def chunk_text(text, max_tokens=400):
    chunks = []
    words = text.split()
    current_chunk_words = []
    for word in words:
        # Predict the tokens to add to the current chunk
        future_chunk = " ".join(current_chunk_words + [word])
        future_tokens = tokenizer.tokenize(future_chunk)
        # If the future chunk would exceed the max_tokens, save the current chunk and start a new one
        if len(future_tokens) + 2 > max_tokens:  # +2 for [CLS] and [SEP]
            chunks.append(" ".join(current_chunk_words))
            current_chunk_words = [word]
        else:
            current_chunk_words.append(word)
    # Don't forget the last chunk
    chunks.append(" ".join(current_chunk_words))
    return chunks

# Your text_prompt here...
text_prompt = """
Intro:
"Hey everyone, welcome back to our channel. Today we're looking at Palantir Technologies' financials to see if it's a buy, hold, or a pass. Don’t forget to hit subscribe for more stock analysis and tips. Let's jump right in."

Content:
"[WOMAN] Palantir has caught the attention of many due to its innovative tech approach. But what about its financial health? Over the last four years, there's been a steady revenue growth, with a 47% increase from 2019 to 2020. However, it wasn’t all positive.

The company saw negative cash flows in 2019 and 2020, but things took a turn in 2021. The Free Cash Flow Yield moved from negative to positive, although it's still below the ideal 7% mark.

Now, onto the margins. The Net Margin improved from a concerning -106.75% in 2020 to -19.61% in 2022. The Earnings Per Share is still negative at -0.02, showing the company isn’t generating positive earnings just yet.

Looking at the balance sheet, the Debt to Equity Ratio sits at 0.319, which is a lower level of debt and that’s good. However, the Price to Book Ratio at 11.64 shows the stock is trading at a premium compared to its book value.

Both the Return on Equity and Return on Assets are in the negative, indicating some inefficiencies in generating income.

Market sentiment is mixed. Some analysts are advising to hold the stock, with a few recent reports having a bearish outlook. The target prices suggested are below the current trading range, which is something to consider.

Despite these financial challenges, Palantir's innovation could potentially give it an edge in the competitive tech sector.

Conclusion:
[WOMAN] So, is Palantir a good investment? It has strengths in revenue growth and innovation, but financial fundamentals and market sentiment suggest a cautious approach. It’s important to weigh the growth prospects against the financial realities before deciding.

That’s all for today’s analysis. Don’t forget to like, share, and subscribe for more insights. Until next time, happy investing!"

Outro:
"[WOMAN] Stay tuned for more stock analysis and remember, invest wisely!"
"""

# Split text into manageable chunks
text_chunks = chunk_text(text_prompt)

# Print chunks to verify
print(f'Number of chunks: {len(text_chunks)}')  # Print the number of chunks
for i, chunk in enumerate(text_chunks):
    print(f'Chunk {i}: {len(tokenizer.tokenize(chunk)) + 2} tokens')  # Print tokens per chunk, +2 for [CLS] and [SEP]

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

print('All audio snippets generated.')
