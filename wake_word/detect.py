import argparse
import numpy as np
import pyaudio
from openwakeword.model import Model

# -------------------------
# Command-line arguments
# -------------------------
parser = argparse.ArgumentParser(description="OpenWakeWord Detector (ONNX)")

parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to your .onnx wakeword model"
)

parser.add_argument(
    "--chunk_size",
    type=int,
    default=1280,
    help="Audio chunk size (default: 1280)"
)

parser.add_argument(
    "--threshold",
    type=float,
    default=0.6,
    help="Wake word detection threshold (default: 0.5)"
)

parser.add_argument(
    "--inference_framework",
    type=str,
    default="onnx",
    choices=["tflite", "onnx"],
    help="Inference framework (default: onnx)"
)

args = parser.parse_args()

# -------------------------
# Audio Settings
# -------------------------
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = args.chunk_size

# -------------------------
# Initialize Microphone
# -------------------------
audio = pyaudio.PyAudio()

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("Loading wakeword model...")

# -------------------------
# Load Model
# -------------------------
model = Model(
    wakeword_models=[args.model_path],
    inference_framework=args.inference_framework
)

print("Model loaded successfully!")
print("=" * 50)
print("Listening...")
print("Press Ctrl+C to stop.")
print("=" * 50)

# -------------------------
# Detection Loop
# -------------------------
try:
    while True:

        pcm = np.frombuffer(
            stream.read(CHUNK, exception_on_overflow=False),
            dtype=np.int16
        )

        predictions = model.predict(pcm)

        for name, score in predictions.items():

            print(f"{name:20s} : {score:.4f}", end="\r")

            if score > args.threshold:
                print("\n")
                print("*" * 50)
                print(f" Wakeword Detected! ({name}) Score={score:.3f}")
                print("*" * 50)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()