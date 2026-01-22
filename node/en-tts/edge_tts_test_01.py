#!/usr/bin/env python3

"""Example showing how to use use .stream() method to get audio chunks
and feed them to SubMaker to generate subtitles"""

import asyncio

import edge_tts

TEXT = "Due to my work. I have long - term communication with overseas partners. almost all through emails.The more I use it, the more I feel that one thing is quite counter - intuitive. Emails seem slow, but in fact. they are very efficient.The efficiency of emails does not lie in quick replies, but in three inherent premises."
VOICE = "en-GB-SoniaNeural"
OUTPUT_FILE = "test.mp3"
SRT_FILE = "test.srt"


async def amain() -> None:
    """Main function"""
    communicate = edge_tts.Communicate(TEXT, VOICE)
    submaker = edge_tts.SubMaker()
    with open(OUTPUT_FILE, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                submaker.feed(chunk)

    with open(SRT_FILE, "w", encoding="utf-8") as file:
        file.write(submaker.get_srt())


if __name__ == "__main__":
    asyncio.run(amain())