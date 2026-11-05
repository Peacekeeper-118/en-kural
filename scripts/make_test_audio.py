"""Generate Tamil / Tanglish test audio with Microsoft neural voices (edge-tts).

Creates a synthetic "my voice" sample set and a "friend's recording" so the
whole pipeline can be exercised without real recordings.  Needs internet.
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from enkural.config import ffmpeg_path  # noqa: E402

# Natural conversational Tamil + Tanglish (English words inside Tamil sentences).
MY_VOICE_TEXTS = [
    "வணக்கம் நண்பர்களே. இன்னைக்கு நாம ஒரு புது topic பத்தி பேசப் போறோம். "
    "வாழ்க்கையில ஏதாவது பண்ணணும்னு நினைச்சா, முதல்ல ஒரு சின்ன step எடுங்க. "
    "பெரிய plan எல்லாம் அப்புறம் வரும்.",
    "நான் college முடிச்ச உடனே job தேடினேன். ரொம்ப கஷ்டமா இருந்துச்சு. "
    "ஆனா ஒவ்வொரு நாளும் கொஞ்சம் கொஞ்சமா முன்னேறினேன். "
    "இப்போ திரும்பிப் பார்த்தா, அந்த struggle எல்லாம் ஒரு நல்ல lesson.",
    "சமையல் பண்ணும்போது patience ரொம்ப முக்கியம். "
    "தீ கொஞ்சம் குறைவா வெச்சு, மசாலா நல்லா வாசனை வரும் வரைக்கும் வதக்கணும். "
    "அப்புறம் தான் தக்காளி சேர்க்கணும்.",
    "வார இறுதியில நாங்க beach போனோம். காத்து நல்லா அடிச்சுச்சு. "
    "சின்ன பசங்க எல்லாம் மணல்ல வீடு கட்டிட்டு இருந்தாங்க. "
    "ரொம்ப நாள் கழிச்சு அப்படி ஒரு relax feeling.",
]
FRIEND_TEXT = (
    "வாழ்க்கையில ஏதாவது பண்ணுவோம்னு decide பண்ணிட்டேன். "
    "இந்த video-ல நான் என்னோட புது journey பத்தி சொல்லப் போறேன். "
    "ஒவ்வொரு நாளும் காலையில எழுந்து, கொஞ்ச நேரம் exercise, அப்புறம் வேலை. "
    "சில நாள் motivation இருக்காது, ஆனா discipline தான் முக்கியம். "
    "நீங்களும் try பண்ணி பாருங்க, comments-ல சொல்லுங்க."
)


async def synth(text: str, voice: str, out_mp3: Path, rate: str = "+0%") -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(str(out_mp3))


def to_wav(src: Path, dst: Path, sr: int = 24000) -> None:
    subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(sr), str(dst)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/data", help="output directory")
    ap.add_argument("--my-voice", default="ta-IN-PallaviNeural", help="voice for the 'my voice' samples")
    ap.add_argument("--friend-voice", default="ta-IN-ValluvarNeural", help="voice for the friend's recording")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    async def run():
        for i, t in enumerate(MY_VOICE_TEXTS, 1):
            mp3 = out / f"my_voice_{i}.mp3"
            await synth(t, args.my_voice, mp3)
            to_wav(mp3, out / f"my_voice_{i}.wav")
            print("wrote", mp3, "and .wav")
        mp3 = out / "friend.mp3"
        await synth(FRIEND_TEXT, args.friend_voice, mp3, rate="+5%")
        to_wav(mp3, out / "friend.wav")
        # an m4a variant to exercise format support
        subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-i", str(mp3), "-c:a", "aac", "-b:a", "128k", str(out / "friend.m4a")], check=True)
        print("wrote", mp3, "friend.wav, friend.m4a")

    asyncio.run(run())


if __name__ == "__main__":
    main()
