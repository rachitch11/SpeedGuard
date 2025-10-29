# generate_beep.py
import numpy as np
import wave

def generate_beep(freq=1000, duration=0.5, filename="beep.wav"):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(freq * t * 2 * np.pi)
    audio = (tone * 32767).astype(np.int16)

    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio.tobytes())

    print(f"{filename} generated!")

if __name__ == "__main__":
    generate_beep()