
import asyncio
import os
import time

import sounddevice as sd
from dotenv import load_dotenv
from cartesia import AsyncCartesia


# ============================================================
# 1. CONFIGURATION
# ============================================================

print("\n[START] Cartesia Sonic-3.6 real-time playback test", flush=True)

load_dotenv()

API_KEY = os.getenv("CARTESIA_API_KEY")
MODEL_ID = os.getenv("CARTESIA_MODEL_ID", "sonic-3.6")

VOICE_ID = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"

SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = "int16"

TEXT_CHUNKS = [
    "Hello! I can help you with your order. ",
    "I will keep you updated with the details."
]

if not API_KEY:
    raise SystemExit(
        "[ERROR] CARTESIA_API_KEY not found. Check your .env file."
    )

print("[CONFIG] API key found (hidden)", flush=True)
print(f"[CONFIG] Model: {MODEL_ID}", flush=True)
print(f"[CONFIG] Voice ID: {VOICE_ID}", flush=True)
print(f"[CONFIG] Sample rate: {SAMPLE_RATE} Hz", flush=True)
print(f"[CONFIG] Text chunks: {len(TEXT_CHUNKS)}", flush=True)
print(
    f"[CONFIG] Total characters: "
    f"{sum(len(chunk) for chunk in TEXT_CHUNKS)}",
    flush=True,
)


# ============================================================
# 2. REAL-TIME TTS
# ============================================================

async def main():
    start_time = time.perf_counter()

    first_text_time = None
    first_audio_time = None
    audio_chunks = 0
    total_audio_bytes = 0
    stream_errors = []

    print("\n[CLIENT] Creating Cartesia client...", flush=True)

    async with AsyncCartesia(
        api_key=API_KEY,
        max_retries=0,
    ) as client:

        print("[CLIENT] Client created.", flush=True)
        print("[WEBSOCKET] Connecting...", flush=True)
        connection_start = time.perf_counter()
        async with client.tts.websocket_connect() as ws:
            connection_end = time.perf_counter()
            connection_time = connection_end - connection_start
            print("[WEBSOCKET] Connected.", flush=True)
            print(
               f"[LATENCY] WebSocket connection established in "
               f"{connection_time:.3f} seconds",
               flush=True,
            )
            print("[TTS] Creating streaming context...", flush=True)

            ctx = ws.context(
                model_id=MODEL_ID,
                voice=VOICE_ID,
                output_format={
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": SAMPLE_RATE,
                },
                language="en",
            )

            print("[TTS] Context ready.", flush=True)

            # ------------------------------------------------
            # 3. OPEN AUDIO DEVICE
            # ------------------------------------------------

            print("[AUDIO] Opening speaker output...", flush=True)

            with sd.RawOutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=0,
            ) as audio_stream:

                print(
                    "[AUDIO] Speaker output ready. "
                    "Audio will play as chunks arrive.",
                    flush=True,
                )

                # --------------------------------------------
                # 4. RECEIVE AND PLAY AUDIO
                # --------------------------------------------

                async def receive_audio():
                    nonlocal first_audio_time
                    nonlocal audio_chunks, total_audio_bytes

                    print("[RECEIVER] Waiting for audio...", flush=True)

                    async for event in ctx.receive():

                        if event.type == "chunk" and event.audio:
                            now = time.perf_counter()

                            if first_audio_time is None:
                                first_audio_time = now

                                print(
                                    f"\n[LATENCY] First audio arrived "
                                    f"{now - first_text_time:.3f}s "
                                    f"after first text was sent.",
                                    flush=True,
                                )

                            audio_chunks += 1
                            chunk_size = len(event.audio)
                            total_audio_bytes += chunk_size

                            # Play this chunk immediately.
                            # No WAV file is created.
                            audio_stream.write(event.audio)

                            print(
                                f"[PLAYBACK] Chunk #{audio_chunks} "
                                f"played: {chunk_size} bytes; "
                                f"total: {total_audio_bytes} bytes",
                                flush=True,
                            )

                        elif event.type == "error":
                            message = (
                                getattr(event, "message", None)
                                or getattr(event, "title", None)
                                or "Unknown streaming error"
                            )

                            stream_errors.append(str(message))

                            print(
                                f"[ERROR] Cartesia: {message}",
                                flush=True,
                            )
                            break

                        elif event.type == "done":
                            print(
                                "[RECEIVER] Cartesia finished generating audio.",
                                flush=True,
                            )
                            break

                    print(
                        "[RECEIVER] Audio receiving completed.",
                        flush=True,
                    )

                # Start receiving before sending text.
                receiver_task = asyncio.create_task(receive_audio())

                # Give the receiver a chance to start.
                await asyncio.sleep(0.1)

                # --------------------------------------------
                # 5. SEND TEXT PROGRESSIVELY
                # --------------------------------------------

                print("\n[TEXT] Starting progressive input...", flush=True)

                first_text_time = time.perf_counter()

                for index, text_chunk in enumerate(TEXT_CHUNKS, start=1):
                    print(
                        f"[TEXT] Sending chunk {index}/"
                        f"{len(TEXT_CHUNKS)}: {text_chunk!r}",
                        flush=True,
                    )

                    await ctx.push(text_chunk)

                    print(
                        f"[TEXT] Chunk {index} sent.",
                        flush=True,
                    )

                    if index < len(TEXT_CHUNKS):
                        # Simulate text arriving from LangGraph.
                        await asyncio.sleep(0.5)

                print(
                    "[TEXT] All text sent. Ending input...",
                    flush=True,
                )

                await ctx.no_more_inputs()

                print(
                    "[TEXT] Input finished. Waiting for final audio...",
                    flush=True,
                )

                await receiver_task

                print(
                    "[AUDIO] Waiting for speaker buffer to finish...",
                    flush=True,
                )

                # Wait for queued speaker audio to play.
                await asyncio.to_thread(audio_stream.stop)

    # ========================================================
    # 6. RESULTS
    # ========================================================

    total_time = time.perf_counter() - start_time

    print("\n========== TEST RESULTS ==========", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Audio chunks played: {audio_chunks}", flush=True)
    print(f"Total audio bytes: {total_audio_bytes}", flush=True)
    print(f"Total elapsed time: {total_time:.3f}s", flush=True)

    if first_audio_time is not None:
        print(
            f"First audio after text started: "
            f"{first_audio_time - first_text_time:.3f}s",
            flush=True,
        )
        print(
            f"First audio after test started: "
            f"{first_audio_time - start_time:.3f}s",
            flush=True,
        )
    else:
        print("[RESULT] No audio chunks received.", flush=True)

    if stream_errors:
        print(f"[RESULT] Errors: {stream_errors}", flush=True)

    print("[RESULT] No WAV file was created.", flush=True)
    print("==================================", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOP] Test cancelled.", flush=True)
    except Exception as error:
        print(
            f"\n[ERROR] {type(error).__name__}: {error}",
            flush=True,
        )
        raise
