
import asyncio
import logging
import os
import json

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, inference
from livekit.plugins import cartesia, silero

from graph import graph, initialize_mcp


# Load the .env located beside this file.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")

server = AgentServer()

CARTESIA_MODEL_ID = os.getenv("CARTESIA_MODEL_ID", "sonic-3.6")
CARTESIA_VOICE_ID = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    await initialize_mcp()

    logger.info("Connected to room: %s", ctx.room.name)

    async def send_voice_event(payload: dict):
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps(payload).encode("utf-8"),
                reliable=True,
                topic="voice-events",
            )
        except Exception:
            logger.exception("Failed to publish voice event")

    session = AgentSession(
        stt=inference.STT(
            model="google/gemini-3.5-transcribe-live",
        ),
        vad=silero.VAD.load(),
        tts=cartesia.TTS(
            api_key=os.getenv("CARTESIA_API_KEY"),
            model=CARTESIA_MODEL_ID,
            voice=CARTESIA_VOICE_ID,
            sample_rate=44100,
        ),
        allow_interruptions=False,
    )

    turn_lock = asyncio.Lock()
    processed_item_ids = set()

    async def process_turn(user_text: str):
        async with turn_lock:
            try:
                await send_voice_event({
                    "type": "transcript",
                    "text": user_text,
                })
                await send_voice_event({
                    "type": "status",
                    "value": "thinking",
                })

                logger.info("Processing transcript: %s", user_text)

                result = await graph.ainvoke(
                    {
                        "message": user_text,
                        "route": "",
                        "response": "",
                        "messages": [],
                    },
                    config={
                        "configurable": {
                            "thread_id": ctx.room.name,
                        }
                    },
                )

                answer = (result.get("response") or "").strip()
                if not answer:
                    answer = (
                        "Sorry, I couldn't get a response. "
                        "Please try again."
                    )

                logger.info("Graph response: %s", answer)

                await send_voice_event({
                    "type": "answer",
                    "text": answer,
                })
                await send_voice_event({
                    "type": "status",
                    "value": "speaking",
                })

                # Do not attempt playback if the session has stopped.
                speech_handle = await session.say(
                    answer,
                    allow_interruptions=False,
                )
                await speech_handle.wait_for_playout()

                await send_voice_event({
                    "type": "status",
                    "value": "listening",
                })
                logger.info("Response playback finished")

            except asyncio.CancelledError:
                logger.info("Voice turn cancelled")
                raise

            except Exception:
                logger.exception("Voice turn failed")

                # Avoid a second failure if the session is no longer running.
                # Send the error status to the frontend instead.
                await send_voice_event({
                    "type": "status",
                    "value": "error",
                })

    @session.on("user_input_transcribed")
    def on_transcript(event):
        user_text = (event.transcript or "").strip()

        logger.info(
            "TRANSCRIPT EVENT: item_id=%r, is_final=%s, text=%r",
            event.item_id,
            event.is_final,
            user_text,
        )

        
        if not user_text:
            return

        # Send partial text to the frontend while the user is speaking.
        if not event.is_final:
            asyncio.create_task(
                send_voice_event({
                    "type": "interim_transcript",
                    "text": user_text,
                })
            )
            return

        # Process only completed utterances.
        if event.item_id is not None:
            if event.item_id in processed_item_ids:
                logger.info(
                    "Ignoring duplicate transcript item: %s",
                    event.item_id,
                )
                return

            processed_item_ids.add(event.item_id)

        asyncio.create_task(process_turn(user_text))

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=(
                "You are a voice interface for an existing assistant. "
                "The application processes requests using its existing "
                "LangGraph workflow. Do not invent business or order data."
            )
        ),
    )

    logger.info("Voice session started")


if __name__ == "__main__":
    agents.cli.run_app(server)
