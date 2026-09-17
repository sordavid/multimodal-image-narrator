from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000

class SpeechNarrator:
    """Converts text into WAV audio file using Kokoro."""

    def __init__(self, language="a", voice="af_heart"):
        self.voice = voice
        self.pipeline = KPipeline(lang_code=language)

    def save(self, text, output_path, speed=1.0):
        """Generate speech from text and save it as a WAV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_chunks = []

        for result in self.pipeline(
            text,
            voice=self.voice,
            speed=speed,
        ):
            if result.audio is not None:
                audio = result.audio.detach().cpu().numpy()
                audio_chunks.append(audio)

        if not audio_chunks:
            raise RuntimeError("Kokoro did not generate any audio")

        combined_audio = np.concatenate(audio_chunks)
        sf.write(output_path, combined_audio, SAMPLE_RATE)

        return output_path