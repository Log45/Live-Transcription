import argparse
import os
import numpy as np
import speech_recognition as sr
import threading
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from faster_whisper import WhisperModel


class TranscriptionWindow:
    def __init__(self, text_queue):
        self.text_queue = text_queue

        self.root = tk.Tk()
        self.root.title("Transcription Display")
        self.root.configure(bg="black")
        self.root.geometry("800x200")
        self.root.resizable(True, True)

        self.text_display = ScrolledText(
            self.root, wrap=tk.WORD, font=("Arial", 16),
            fg="white", bg="black", padx=10, pady=10
        )
        self.text_display.pack(expand=True, fill="both")

        self.update_text()
        self.root.mainloop()

    def update_text(self):
        while not self.text_queue.empty():
            new_word = self.text_queue.get_nowait()
            self.text_display.insert(tk.END, new_word + " ")
            self.text_display.see(tk.END)
        self.root.after(200, self.update_text)


def run_whisper(text_queue, args):
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    if "linux" in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == "list":
            print("Available microphones:")
            for idx, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"  {idx}: {name}")
            return
        else:
            found_idx = None
            for idx, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    found_idx = idx
                    break
            if found_idx is None:
                print(f"ERROR: Microphone '{mic_name}' not found.")
                return
            source = sr.Microphone(sample_rate=16000, device_index=found_idx)
    else:
        source = sr.Microphone(sample_rate=16000)

    print(f"Loading faster-whisper model: {args.model} ...")
    model = WhisperModel(args.model, compute_type="int8")  # int8 = fastest
    print("Model loaded.\n")

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    phrase_time = None
    data_queue = Queue()

    def record_callback(_, audio: sr.AudioData):
        data = audio.get_raw_data()
        data_queue.put(data)

    with source:
        print("Calibrating mic for ambient noise...")
        recorder.adjust_for_ambient_noise(source, duration=1)
        print("Calibration complete. Starting background listening...")

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    print("Recording... Speak!\n")

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                chunk_gap = False
                if phrase_time and (now - phrase_time) > timedelta(seconds=phrase_timeout):
                    chunk_gap = True
                phrase_time = now

                audio_data = b"".join(list(data_queue.queue))
                data_queue.queue.clear()

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                segments, _ = model.transcribe(audio_np, language="en", word_timestamps=True)

                if chunk_gap:
                    text_queue.put("\n")

                for segment in segments:
                    for word_info in segment.words:
                        word = word_info.word.strip()
                        print(word, end=" ", flush=True)
                        text_queue.put(word)

            else:
                sleep(0.25)

        except KeyboardInterrupt:
            break

    print("\n\nFinal Transcript:")
    while not text_queue.empty():
        print(text_queue.get(), end=" ")


def main():
    parser = argparse.ArgumentParser(description="Real-time mic transcription with faster-whisper on Raspberry Pi.")
    parser.add_argument("--model", default="tiny.en", choices=["tiny", "tiny.en", "base", "base.en"], help="Faster-Whisper model to use.")
    parser.add_argument("--non_english", action="store_true", help="Keep for compatibility. Ignored in faster-whisper.")
    parser.add_argument("--energy_threshold", default=1000, type=int, help="Mic detection threshold.")
    parser.add_argument("--record_timeout", default=1.0, type=float, help="Seconds of audio before processing.")
    parser.add_argument("--phrase_timeout", default=1.0, type=float, help="Silence gap between phrases.")

    if "linux" in platform:
        parser.add_argument("--default_microphone", default="pulse", type=str, help="Mic name on Linux. Use 'list' to list devices.")

    args = parser.parse_args()

    text_queue = Queue()

    whisper_thread = threading.Thread(target=run_whisper, args=(text_queue, args), daemon=True)
    whisper_thread.start()

    TranscriptionWindow(text_queue)


if __name__ == "__main__":
    main()
