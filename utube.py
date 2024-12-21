import os
import re
import whisper
import streamlit as st
from transformers import pipeline
import subprocess

st.title('Get Your Answers')

# Helper function to validate YouTube URLs
def is_valid_youtube_url(url):
    youtube_regex = r"(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+"
    return re.match(youtube_regex, url) is not None

# Helper function to download audio using yt-dlp
def download_audio(youtube_url):
    try:
        output_file = "audio.mp3"
        subprocess.run([
            "yt-dlp",
            "-x", "--audio-format", "mp3",
            "--output", output_file,
            youtube_url
        ], check=True)
        return output_file
    except subprocess.CalledProcessError as e:
        return f"Error: {str(e)}"

# Helper function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        return f"Error: {str(e)}"

# Helper function to summarize text using Hugging Face
def summarize_text(transcript):
    try:
        summarizer = pipeline("summarization")
        summary = summarizer(transcript, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {str(e)}"

# Helper function to answer questions using Hugging Face
def answer_question(transcript, question):
    try:
        question_answerer = pipeline("question-answering")
        answer = question_answerer(question=question, context=transcript)
        detailed_answer = f" {answer['answer']}."
        return detailed_answer
    except Exception as e:
        return f"Error: {str(e)}"

# Input field for YouTube URL
url = st.text_input("Paste YouTube Link:")

# Process if a valid URL is entered
if url:
    if not is_valid_youtube_url(url):
        st.error("Invalid YouTube URL. Please try again.")
    else:
        # Store transcript to avoid redoing the process
        if "transcript.txt" not in os.listdir():
            with st.spinner("Downloading audio..."):
                audio_path = download_audio(url)
                if "Error" in audio_path:
                    st.error(audio_path)
                else:
                    st.success("Audio downloaded successfully!")

                    # Transcribe the audio
                    with st.spinner("Transcribing audio..."):
                        transcript = transcribe_audio(audio_path)
                        if "Error" in transcript:
                            st.error(transcript)
                        else:
                            st.success("Transcription completed!")
                            with open("transcript.txt", "w") as file:
                                file.write(transcript)
                            st.text_area("Transcript:", transcript, height=200)

        # Load the transcript from the saved file
        else:
            with open("transcript.txt", "r") as file:
                transcript = file.read()
            st.text_area("Transcript:", transcript, height=200)

        # Summarize the transcript
        if st.button("Summarize Transcript"):
            with st.spinner("Generating summary..."):
                summary = summarize_text(transcript)
                if "Error" in summary:
                    st.error(summary)
                else:
                    st.success("Summary completed!")
                    st.text_area("Summary:", summary, height=150)

        # Answer questions about the transcript
        question = st.text_input("Ask a question about the transcript:")
        if question:
            if st.button("Get Answer"):
                with st.spinner("Fetching answer..."):
                    answer = answer_question(transcript, question)
                    if "Error" in answer:
                        st.error(answer)
                    else:
                        st.success("Answer fetched!")
                        st.write("Answer:", answer)
