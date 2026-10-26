
import { useRef, useState } from "react";
import { Room, RoomEvent, Track } from "livekit-client";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function createId() {
  return crypto.randomUUID();
}

function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState("");
  const [speaking, setSpeaking] = useState(false);
  const [stoppingVoice, setStoppingVoice] = useState(false);

  const [threadId] = useState(() => crypto.randomUUID());

  const liveKitRoomRef = useRef(null);
  const liveKitAudioElementsRef = useRef([]);
  const transcriptRef = useRef("");
  const voiceTurnRef = useRef({
    userId: null,
    assistantId: null,
  });

  const liveKitTurnActiveRef = useRef(false);
  const stopRequestedRef = useRef(false);
  const textRequestActiveRef = useRef(false);

  function updateMessage(messageId, update) {
    if (!messageId) return;

    setMessages((previous) =>
      previous.map((item) => {
        if (item.id !== messageId) return item;

        return typeof update === "function"
          ? update(item)
          : { ...item, ...update };
      }),
    );
  }

  function updateVoiceUserMessage(text) {
    updateMessage(voiceTurnRef.current.userId, {
      content: text,
    });
  }

  function updateVoiceAssistantMessage(update) {
    updateMessage(voiceTurnRef.current.assistantId, update);
  }

  function createLiveKitVoiceTurn() {
    const userId = createId();
    const assistantId = createId();

    voiceTurnRef.current = { userId, assistantId };
    liveKitTurnActiveRef.current = true;

    setMessages((previous) => [
      ...previous,
      {
        id: userId,
        role: "user",
        content: "Listening...",
        status: "recording",
      },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        status: "idle",
      },
    ]);
  }

  function removeRemoteAudioElements() {
    liveKitAudioElementsRef.current.forEach((element) => {
      element.pause();
      element.srcObject = null;
      element.remove();
    });

    liveKitAudioElementsRef.current = [];
  }

  function resetVoiceState() {
    liveKitRoomRef.current = null;
    liveKitTurnActiveRef.current = false;
    stopRequestedRef.current = false;

    setRecording(false);
    setLoading(false);
    setSpeaking(false);
    setStoppingVoice(false);
    setLiveTranscript("");

    transcriptRef.current = "";
    removeRemoteAudioElements();
  }

  async function disconnectLiveKitRoom() {
    const room = liveKitRoomRef.current;

    liveKitRoomRef.current = null;

    if (room) {
      await room.disconnect();
    }

    resetVoiceState();
  }

  async function startLiveKitVoice() {
    if (liveKitRoomRef.current || stoppingVoice) return;

    setLiveTranscript("");
    transcriptRef.current = "";
    stopRequestedRef.current = false;

    createLiveKitVoiceTurn();

    let room;

    try {
      setLoading(true);

      const response = await fetch(`${API_URL}/livekit/token`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error(`Token request failed: ${response.status}`);
      }

      const { serverUrl, participantToken } = await response.json();

      room = new Room({
        adaptiveStream: true,
        dynacast: true,
      });

      liveKitRoomRef.current = room;

      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind !== Track.Kind.Audio) return;

        const audioElement = track.attach();
        audioElement.autoplay = true;
        document.body.appendChild(audioElement);

        liveKitAudioElementsRef.current.push(audioElement);

        audioElement.play().catch((error) => {
          console.warn("Could not autoplay remote audio:", error);
        });
      });

      room.on(RoomEvent.TrackUnsubscribed, (track) => {
        track.detach().forEach((element) => {
          element.pause();
          element.srcObject = null;
          element.remove();

          liveKitAudioElementsRef.current =
            liveKitAudioElementsRef.current.filter(
              (item) => item !== element,
            );
        });
      });

      room.on(RoomEvent.DataReceived, async (data, participant, kind, topic) => {
        if (topic !== "voice-events") return;

        try {
          const event = JSON.parse(new TextDecoder().decode(data));

           console.log("LiveKit voice event:", event);

          if (
            event.type === "interim_transcript" ||
            event.type === "transcript"
          ) {
            if (!liveKitTurnActiveRef.current) {
              createLiveKitVoiceTurn();
            }

            const text = event.text || "";

            transcriptRef.current = text;
            setLiveTranscript(text);
            updateVoiceUserMessage(text);
          }

          if (event.type === "status") {
            if (event.value === "thinking") {
              setLoading(true);
              setSpeaking(false);
            }

            if (event.value === "speaking") {
              setLoading(false);
              setSpeaking(true);
            }

            if (event.value === "listening") {
              setLoading(false);
              setSpeaking(false);
              setLiveTranscript("");

              if (stopRequestedRef.current) {
                await disconnectLiveKitRoom();
                return;
              }

              await room.localParticipant.setMicrophoneEnabled(true);
              setRecording(true);
            }

            if (event.value === "error") {
              updateVoiceAssistantMessage({
                content: "Sorry, something went wrong. Please try again.",
                status: "complete",
              });

              setLoading(false);
              setSpeaking(false);
              liveKitTurnActiveRef.current = false;

              if (stopRequestedRef.current) {
                await disconnectLiveKitRoom();
              } else {
                await room.localParticipant.setMicrophoneEnabled(true);
                setRecording(true);
              }
            }
          }

          if (event.type === "answer") {
            updateVoiceAssistantMessage({
              content:
                event.text ||
                "I couldn't generate a response. Please try again.",
              status: "complete",
            });

            liveKitTurnActiveRef.current = false;
            setLoading(false);
          }

          if (event.type === "error") {
            updateVoiceAssistantMessage({
              content:
                event.message ||
                "Sorry, something went wrong. Please try again.",
              status: "complete",
            });

            liveKitTurnActiveRef.current = false;
            setLoading(false);
            setSpeaking(false);

            if (stopRequestedRef.current) {
              await disconnectLiveKitRoom();
            }
          }
        } catch (error) {
          console.error("Could not process LiveKit event:", error);
        }
      });

      room.on(RoomEvent.Disconnected, () => {
        resetVoiceState();
      });

      await room.connect(serverUrl, participantToken);
      await room.startAudio();
      await room.localParticipant.setMicrophoneEnabled(true);

      setRecording(true);
      setLoading(false);

      console.log("Connected to LiveKit room:", room.name);
    } catch (error) {
      console.error("LiveKit connection failed:", error);

      if (room) {
        await room.disconnect().catch(() => {});
      }

      resetVoiceState();

      updateVoiceAssistantMessage({
        content: `Could not start voice: ${error.message}`,
        status: "complete",
      });
    }
  }

  
  async function stopLiveKitVoice() {
    const room = liveKitRoomRef.current;

    if (!room) return;

    stopRequestedRef.current = true;
    setStoppingVoice(true);
    setRecording(false);
    setLoading(false);
    setSpeaking(false);

    try {
      // Disconnect immediately instead of waiting for the response.
      await disconnectLiveKitRoom();
    } catch (error) {
      console.error("Could not stop the LiveKit session:", error);
      resetVoiceState();
    }
  }


  async function sendMessage() {
    const userMessage = message.trim();

    if (
      !userMessage ||
      loading ||
      recording ||
      speaking ||
      stoppingVoice ||
      textRequestActiveRef.current
    ) {
      return;
    }

    const userId = createId();
    const assistantId = createId();

    textRequestActiveRef.current = true;

    setMessages((previous) => [
      ...previous,
      {
        id: userId,
        role: "user",
        content: userMessage,
        status: "complete",
      },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        status: "thinking",
      },
    ]);

    setMessage("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          message: userMessage,
          thread_id: threadId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      if (!response.body) {
        throw new Error("Streaming response is not available.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      const streamStartedAt = performance.now();
      let firstTokenReceived = false;
      let tokenCount = 0;
      let buffer = "";
      let completeAnswer = "";
      let streamError = null;

      function handleSSEEvent(eventText) {
        const dataLine = eventText
          .split(/\r?\n/)
          .find((line) => line.startsWith("data:"));

        if (!dataLine) return;

        let payload;

        try {
          payload = JSON.parse(dataLine.slice(5).trim());
        } catch {
          return;
        }

        if (payload.type === "token") {
          tokenCount++;

          if (!firstTokenReceived) {
            firstTokenReceived = true;

            console.log(
              `First token received: ${(
                (performance.now() - streamStartedAt) /
                1000
              ).toFixed(3)} seconds`,
            );
          }

          completeAnswer += payload.text;

          updateMessage(assistantId, {
            content: completeAnswer,
            status: "streaming",
          });
        }

        if (payload.type === "done") {
          completeAnswer = payload.answer ?? completeAnswer;

          updateMessage(assistantId, {
            content: completeAnswer,
            status: "complete",
          });

          console.log(
            `Total frontend streaming time: ${(
              (performance.now() - streamStartedAt) /
              1000
            ).toFixed(3)} seconds`,
          );

          console.log(
            `Backend streaming completed in ${payload.elapsed_seconds} seconds`,
          );
        }

        if (payload.type === "error") {
          streamError = payload.message || "The server returned an error.";
        }
      }

      while (true) {
        const { value, done } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split(/\r?\n\r?\n/);
        buffer = events.pop() || "";

        for (const eventText of events) {
          handleSSEEvent(eventText);
        }
      }

      buffer += decoder.decode();

      if (buffer.trim()) {
        handleSSEEvent(buffer);
      }

      if (streamError) {
        throw new Error(streamError);
      }

      if (!completeAnswer.trim()) {
        throw new Error("The server returned no answer.");
      }
    } catch (error) {
      console.error("Streaming chat failed:", error);

      updateMessage(assistantId, {
        content:
          error.message || "Sorry, something went wrong. Please try again.",
        status: "complete",
      });
    } finally {
      textRequestActiveRef.current = false;
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <main className="chat-container">
        <header className="chat-header">
          <div className="brand">
            <div className="brand-icon">🛍️</div>
            <div>
              <h1>Commerce Assistant</h1>
              <p>Your shopping support, made simple</p>
            </div>
          </div>

          <div className="online-status">
            <span className="online-dot" />
            Online
          </div>
        </header>

        <section className="messages" aria-live="polite">
          {messages.length === 0 && (
            <div className="welcome-panel">
              <div className="welcome-icon">✨</div>
              <h2>How can I help you today?</h2>
              <p>
                Ask about orders, shipping, returns, products, and more. You
                can also use the microphone to speak.
              </p>

              <div className="suggestion-list">
                <button
                  type="button"
                  onClick={() => setMessage("What is your return policy?")}
                  disabled={loading || recording || stoppingVoice}
                >
                  Return policy
                </button>

                <button
                  type="button"
                  onClick={() => setMessage("How can I track my order?")}
                  disabled={loading || recording || stoppingVoice}
                >
                  Track an order
                </button>

                <button
                  type="button"
                  onClick={() => setMessage("How long does shipping take?")}
                  disabled={loading || recording || stoppingVoice}
                >
                  Shipping information
                </button>
              </div>
            </div>
          )}

          {messages
            .filter(
              (msg) => !(msg.role === "assistant" && msg.status === "idle"),
            )
            .map((msg) => (
              <div key={msg.id} className={`message-row ${msg.role}`}>
                {msg.role === "assistant" && (
                  <div className="avatar assistant-avatar">✦</div>
                )}

                <div
                  className={`message ${msg.role} ${
                    msg.status === "thinking" ? "thinking-message" : ""
                  }`}
                >
                  {msg.role === "assistant" && msg.status === "thinking" ? (
                    <div className="thinking-content">
                      <span className="thinking-dots" aria-hidden="true">
                        <i />
                        <i />
                        <i />
                      </span>
                      <span>Thinking...</span>
                    </div>
                  ) : msg.role === "assistant" && !msg.content ? (
                    msg.status === "idle" ? null : (
                      <div className="thinking-content">
                        <span className="thinking-dots" aria-hidden="true">
                          <i />
                          <i />
                          <i />
                        </span>
                        <span>Preparing your answer...</span>
                      </div>
                    )
                  ) : (
                    <div className="message-content">
                      {msg.content}
                      {msg.role === "assistant" &&
                        msg.status === "streaming" && (
                          <span
                            className="streaming-cursor"
                            aria-hidden="true"
                          />
                        )}
                    </div>
                  )}
                </div>

                {msg.role === "user" && (
                  <div className="avatar user-avatar">You</div>
                )}
              </div>
            ))}

          {recording && (
            <div className="recording-status">
              <span className="recording-pulse" />
              Listening — speak naturally, then press Stop.
              {liveTranscript && (
                <span className="transcript-preview">{liveTranscript}</span>
              )}
            </div>
          )}

          {stoppingVoice && (
            <div className="recording-status">
              Finishing your request. Please wait for the assistant...
            </div>
          )}

          {speaking && (
            <div className="audio-status">
              <span className="audio-bars" aria-hidden="true">
                <i />
                <i />
                <i />
                <i />
                <i />
              </span>
              Assistant is speaking...
            </div>
          )}
        </section>

        <footer className="input-area">
          <form
            className="message-form"
            onSubmit={(event) => {
              event.preventDefault();
              sendMessage();
            }}
          >
            <input
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              placeholder="Ask about orders, returns, shipping..."
              disabled={
                loading || recording || speaking || stoppingVoice
              }
              aria-label="Type your message"
            />

            <button
              className="send-button"
              type="submit"
              disabled={
                !message.trim() ||
                loading ||
                recording ||
                speaking ||
                stoppingVoice
              }
            >
              {loading ? "Waiting..." : "Send"}
              <span aria-hidden="true">➤</span>
            </button>
          </form>

          <div className="voice-controls">
            {!recording && !stoppingVoice ? (
              <button
                className="voice-button start-button"
                type="button"
                onClick={startLiveKitVoice}
                disabled={loading || speaking}
              >
                <span aria-hidden="true">🎙️</span>
                Start voice
              </button>
            ) : recording ? (
              <button
                className="voice-button stop-button"
                type="button"
                onClick={stopLiveKitVoice}
              >
                <span className="stop-square" />
                Stop and send
              </button>
            ) : (
              <button
                className="voice-button stop-button"
                type="button"
                disabled
              >
                Finishing...
              </button>
            )}

            <span className="input-hint">
              {recording
                ? "Your transcript stays visible after you stop."
                : stoppingVoice
                  ? "Waiting for the assistant to finish."
                  : loading
                    ? "Your assistant is working on your request."
                    : "Type a message or use voice input."}
            </span>
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
