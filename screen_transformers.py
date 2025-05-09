import numpy as np
from pathlib import Path
from optimum.onnxruntime import (
    AutoQuantizationConfig,
    ORTModelForSpeechSeq2Seq,
    ORTQuantizer
)
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor
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

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

def quantize_model(model_id="medium", save_dir="quantized_whisper", use_4bit=True):
    """
    Quantizes Whisper model for CPU using dynamic int8 quantization and saves it.

    Args:
        model_id (str): Size of Whisper model (e.g., 'tiny', 'base', 'medium').
        save_dir (str): Directory to save the quantized model.
    """
    model_name = f"openai/whisper-{model_id}"

    # Load the original model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.eval()

    # Quantize the model dynamically (int8 for CPU)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        dtype=torch.qint8,  # Using int8 quantization for CPU
        inplace=False
    )

    # Save quantized model
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    quantized_model.save_pretrained(save_dir)

    # Save processor too
    processor = AutoProcessor.from_pretrained(model_name)
    processor.save_pretrained(save_dir)

    print(f"Quantized model saved to: {save_dir}")

def load_model(model_dir):
    # Load the model and processor
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, device_map="auto")
    return processor, model

@torch.inference_mode()
def transcribe(processor, model, audio_data: np.ndarray):
    model.eval()

    # Prepare input features
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt").to(model.device)

    # Generate
    
    generated_ids = model.generate(inputs["input_features"])

    # Decode
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

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
        source = sr.Microphone(1, sample_rate=16000)

    print(f"Loading whisper model: {args.model} ...")

    processor, model = load_model(args.model)
    
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

                segments = transcribe(processor, model, audio_np)

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
    parser = argparse.ArgumentParser(description="Real-time mic transcription with huggingface whisper on Raspberry Pi.")
    parser.add_argument("--model", default="small.en", choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "large", "medium", "medium.en"], help="Faster-Whisper model to use.")
    parser.add_argument("--non_english", action="store_true", help="Keep for compatibility. Ignored in faster-whisper.")
    parser.add_argument("--energy_threshold", default=1000, type=int, help="Mic detection threshold.")
    parser.add_argument("--record_timeout", default=1.0, type=float, help="Seconds of audio before processing.")
    parser.add_argument("--phrase_timeout", default=1.0, type=float, help="Silence gap between phrases.")
    parser.add_argument("--quantize", action="store_true", help="Quantize the model.")
    parser.add_argument("--quant_precision", default="int4", choices = ["int4", "int2", "int8", "float8"], type=str, help="Quantization precision.")

    if "linux" in platform:
        parser.add_argument("--default_microphone", default="pulse", type=str, help="Mic name on Linux. Use 'list' to list devices.")

    args = parser.parse_args()

    if args.quantize:
        model_id = args.model
        save_dir = f"quantized_{model_id}"
        quantize_model(model_id, save_dir)
        print(f"Quantized model saved to {save_dir}")
        return
    else:
        if os.path.exists(f"quantized_{args.model}"):
            print(f"Loading quantized model from quantized_{args.model}")
            args.model = f"quantized_{args.model}"
        else:
            print(f"Quantized model not found. Please run `screen.py` to utilize an non-quantized model. Or run `--quantize` to quantize the model.")
            return

    text_queue = Queue()

    whisper_thread = threading.Thread(target=run_whisper, args=(text_queue, args), daemon=True)
    whisper_thread.start()

    TranscriptionWindow(text_queue)


if __name__ == "__main__":
    main()