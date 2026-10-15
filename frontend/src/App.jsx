import { useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000";

function createId() {
  return crypto.randomUUID();
}

function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState("");
  const [audioPlaying, setAudioPlaying] = useState(false);
  const [speaking, setSpeaking] = useState(false);

  const [threadId] = useState(() => crypto.randomUUID());

  const socketRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const playbackAudioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const gainRef = useRef(null);

  const activeAudioSourcesRef = useRef(new Set());
  const ttsFinishedRef = useRef(false);
  const nextPlayTimeRef = useRef(0);
  const audioQueueRef = useRef(Promise.resolve());
  const playbackSampleRateRef = useRef(44100);

  const transcriptRef = useRef("");
  const voiceTurnRef = useRef({
    userId: null,
    assistantId: null,
  });

  const voiceStartedRef = useRef(false);
  const textRequestActiveRef = useRef(false);

  const voiceTimingRef = useRef({
    startedAt: 0,
    firstChunkReceived: false,
    firstPlaybackStarted: false,
  });

  // Update one specific message. Never assume the last message
  // belongs to the request currently producing streamed text.
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

  // ==========================================================
  // AUDIO CONVERSION: FLOAT32 -> PCM INT16
  // ==========================================================

  function convertToPCM(input, inputSampleRate) {
    const outputSampleRate = 16000;
    const ratio = inputSampleRate / outputSampleRate;
    const outputLength = Math.floor(input.length / ratio);
    const downsampled = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const start = Math.floor(i * ratio);
      const end = Math.min(Math.floor((i + 1) * ratio), input.length);

      let sum = 0;
      let count = 0;

      for (let j = start; j < end; j++) {
        sum += input[j];
        count++;
      }

      downsampled[i] = count > 0 ? sum / count : 0;
    }

    const pcm = new Int16Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const sample = Math.max(-1, Math.min(1, downsampled[i]));
      pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }

    return pcm;
  }

  // ==========================================================
  // CLEAN UP MICROPHONE
  // ==========================================================

  async function cleanupMicrophone() {
    if (processorRef.current) {
      processorRef.current.onaudioprocess = null;

      try {
        processorRef.current.disconnect();
      } catch {
        // Already disconnected.
      }

      processorRef.current = null;
    }

    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch {
        // Already disconnected.
      }

      sourceRef.current = null;
    }

    if (gainRef.current) {
      try {
        gainRef.current.disconnect();
      } catch {
        // Already disconnected.
      }

      gainRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      const context = audioContextRef.current;
      audioContextRef.current = null;

      if (context.state !== "closed") {
        await context.close();
      }
    }
  }

  // ==========================================================
  // PLAY CARTESIA AUDIO
  // ==========================================================

  async function playAudioChunk(arrayBuffer) {
    let audioContext = playbackAudioContextRef.current;

    if (!audioContext || audioContext.state === "closed") {
      audioContext = new AudioContext();
      playbackAudioContextRef.current = audioContext;
    }

    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    const pcm = new Int16Array(arrayBuffer);
    if (pcm.length === 0) return;

    const audioBuffer = audioContext.createBuffer(
      1,
      pcm.length,
      playbackSampleRateRef.current,
    );

    const channel = audioBuffer.getChannelData(0);

    for (let i = 0; i < pcm.length; i++) {
      channel[i] = pcm[i] / (pcm[i] < 0 ? 32768 : 32767);
    }

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    const startAt = Math.max(
      audioContext.currentTime + 0.02,
      nextPlayTimeRef.current,
    );

    activeAudioSourcesRef.current.add(source);

    source.onended = () => {
      activeAudioSourcesRef.current.delete(source);

      if (ttsFinishedRef.current && activeAudioSourcesRef.current.size === 0) {
        setSpeaking(false);
        setAudioPlaying(false);
      }
    };

    source.start(startAt);

    const timing = voiceTimingRef.current;

    if (!timing.firstPlaybackStarted) {
      timing.firstPlaybackStarted = true;

      const delayMs = Math.max(0, (startAt - audioContext.currentTime) * 1000);

      console.log(
        `First chunk scheduled for playback: ${(
          (performance.now() - timing.startedAt + delayMs) /
          1000
        ).toFixed(3)} seconds after tts_start`,
      );
    }

    nextPlayTimeRef.current = startAt + audioBuffer.duration;
    setAudioPlaying(true);
  }

  // ==========================================================
  // HANDLE VOICE WEBSOCKET EVENTS
  // ==========================================================

  function handleVoiceMessage(event) {
    // Binary audio chunk.
    if (event.data instanceof ArrayBuffer) {
      const timing = voiceTimingRef.current;

      if (!timing.firstChunkReceived) {
        timing.firstChunkReceived = true;

        console.log(
          `First audio chunk received: ${(
            (performance.now() - timing.startedAt) /
            1000
          ).toFixed(3)} seconds after tts_start`,
        );
      }

      audioQueueRef.current = audioQueueRef.current
        .then(() => playAudioChunk(event.data))
        .catch((error) => {
          console.error("Audio playback failed:", error);
        });

      return;
    }

    let data;

    try {
      data = JSON.parse(event.data);
    } catch {
      return;
    }

    if (data.type === "ready") {
      // The ready event is handled by the ready promise below.
      return;
    }

    if (data.type === "tts_start") {
      playbackSampleRateRef.current = data.sample_rate || 44100;

      voiceTimingRef.current = {
        startedAt: performance.now(),
        firstChunkReceived: false,
        firstPlaybackStarted: false,
      };

      console.log("Cartesia TTS started");

      setSpeaking(true);
      setAudioPlaying(false);
      nextPlayTimeRef.current = 0;
      ttsFinishedRef.current = false;
      return;
    }

    if (data.type === "tts_end") {
      console.log(
        `Backend finished sending TTS: ${(
          (performance.now() - voiceTimingRef.current.startedAt) /
          1000
        ).toFixed(3)} seconds after tts_start`,
      );

      audioQueueRef.current = audioQueueRef.current.then(() => {
        ttsFinishedRef.current = true;

        if (activeAudioSourcesRef.current.size === 0) {
          setSpeaking(false);
          setAudioPlaying(false);
        }
      });

      return;
    }

    if (data.type === "tts_error") {
      console.error("TTS error:", data.message);
      setSpeaking(false);
      return;
    }

    // Live transcription updates the existing user message.
    if (data.type === "interim") {
      const text = data.text || "";

      transcriptRef.current = text;
      setLiveTranscript(text);

      updateVoiceUserMessage(text || "Listening...");
      return;
    }

    // Final transcription stays visible after Stop is clicked.
    if (data.type === "final") {
      const text = (data.text || "").trim();

      if (text) {
        transcriptRef.current = text;
        setLiveTranscript(text);
        updateVoiceUserMessage(text);
      }

      return;
    }

    // Stream answer chunks into this voice turn's assistant message.
    if (data.type === "answer_chunk") {
      const assistantId = voiceTurnRef.current.assistantId;

      updateMessage(assistantId, (item) => ({
        ...item,
        content: (item.content || "") + (data.text || ""),
        status: "streaming",
      }));

      return;
    }

    // Final answer replaces the streamed draft, avoiding duplication.
    if (data.type === "answer") {
      const transcript = (
        data.transcription ||
        transcriptRef.current ||
        ""
      ).trim();

      if (transcript) {
        transcriptRef.current = transcript;
        updateVoiceUserMessage(transcript);
      }

      updateVoiceAssistantMessage({
        content: data.answer || "",
        status: "complete",
      });

      setLiveTranscript("");
      setLoading(false);
      voiceStartedRef.current = false;
      return;
    }

    if (data.type === "error") {
      updateVoiceAssistantMessage({
        content:
          data.message || "Sorry, something went wrong. Please try again.",
        status: "complete",
      });

      setLoading(false);
      setRecording(false);
      setLiveTranscript("");
      voiceStartedRef.current = false;
    }
  }

  // ==========================================================
  // START VOICE INPUT
  // ==========================================================

  async function startRecording() {
    if (
      recording ||
      loading ||
      speaking ||
      audioPlaying ||
      textRequestActiveRef.current ||
      voiceStartedRef.current
    ) {
      return;
    }

    let userId;
    let assistantId;

    try {
      transcriptRef.current = "";
      setLiveTranscript("");

      if (
        !playbackAudioContextRef.current ||
        playbackAudioContextRef.current.state === "closed"
      ) {
        playbackAudioContextRef.current = new AudioContext();
      }

      await playbackAudioContextRef.current.resume();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      streamRef.current = stream;

      const socket = new WebSocket(
        `${WS_URL}/voice-stream?thread_id=${encodeURIComponent(threadId)}`,
      );

      socket.binaryType = "arraybuffer";
      socketRef.current = socket;

      let resolveReady;
      let rejectReady;
      let readyReceived = false;

      const readyPromise = new Promise((resolve, reject) => {
        resolveReady = resolve;
        rejectReady = reject;
      });

      socket.onmessage = (event) => {
        if (typeof event.data !== "string") {
          handleVoiceMessage(event);
          return;
        }

        let data;

        try {
          data = JSON.parse(event.data);
        } catch {
          return;
        }

        if (data.type === "ready") {
          readyReceived = true;
          resolveReady();
          return;
        }

        handleVoiceMessage(event);
      };

      socket.onerror = () => {
        if (!readyReceived) {
          rejectReady(new Error("WebSocket connection failed."));
        }
      };

      socket.onclose = () => {
        setRecording(false);

        // Do not leave a thinking indicator stuck if the connection closes.
        if (voiceStartedRef.current) {
          updateVoiceAssistantMessage((item) => ({
            ...item,
            content:
              item.content ||
              "The connection closed before the response was completed. Please try again.",
            status: "complete",
          }));

          setLoading(false);
          voiceStartedRef.current = false;
        }
      };

      await new Promise((resolve, reject) => {
        socket.addEventListener("open", resolve, { once: true });
        socket.addEventListener(
          "error",
          () => reject(new Error("Could not connect to FastAPI.")),
          { once: true },
        );
      });

      await readyPromise;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      await audioContext.resume();

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0;
      gainRef.current = silentGain;

      processor.onaudioprocess = (event) => {
        if (socket.readyState !== WebSocket.OPEN) return;

        const input = event.inputBuffer.getChannelData(0);
        const pcm = convertToPCM(input, audioContext.sampleRate);

        if (pcm.length > 0) {
          socket.send(pcm.buffer);
        }
      };

      source.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(audioContext.destination);

      // Create a distinct conversation turn for this voice request.
      userId = createId();
      assistantId = createId();

      voiceTurnRef.current = { userId, assistantId };
      voiceStartedRef.current = true;

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

      setRecording(true);
    } catch (error) {
      console.error("Could not start voice streaming:", error);

      await cleanupMicrophone();

      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }

      setRecording(false);
      setLoading(false);
      voiceStartedRef.current = false;

      setMessages((previous) => [
        ...previous,
        {
          id: createId(),
          role: "assistant",
          content:
            "Could not start voice input. Please check microphone permission and backend connection.",
          status: "complete",
        },
      ]);
    }
  }

  // ==========================================================
  // STOP VOICE INPUT
  // ==========================================================

  async function stopRecording() {
    if (!recording) return;

    setRecording(false);
    setLoading(true);

    // Preserve the last transcript already shown while recording.
    updateVoiceUserMessage(
      transcriptRef.current || liveTranscript || "Voice message recorded",
    );

    // Show Thinking... in the assistant bubble for this same turn.
    updateVoiceAssistantMessage({
      content: "",
      status: "thinking",
    });

    await cleanupMicrophone();

    const socket = socketRef.current;

    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "stop" }));
    } else {
      updateVoiceAssistantMessage({
        content: "The voice connection was lost. Please try again.",
        status: "complete",
      });

      setLoading(false);
      voiceStartedRef.current = false;
    }
  }

  // ==========================================================
  // NORMAL TEXT CHAT - STREAMING
  // ==========================================================

  async function sendMessage() {
    const userMessage = message.trim();

    if (
      !userMessage ||
      loading ||
      recording ||
      speaking ||
      audioPlaying ||
      textRequestActiveRef.current ||
      voiceStartedRef.current
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

          console.log(
            `Token ${tokenCount} received: ${(
              (performance.now() - streamStartedAt) /
              1000
            ).toFixed(3)} seconds`,
            payload.text,
          );

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

  // ==========================================================
  // UI
  // ==========================================================

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
                Ask about orders, shipping, returns, products, and more. You can
                also use the microphone to speak.
              </p>

              <div className="suggestion-list">
                <button
                  type="button"
                  onClick={() => setMessage("What is your return policy?")}
                  disabled={loading || recording}
                >
                  Return policy
                </button>

                <button
                  type="button"
                  onClick={() => setMessage("How can I track my order?")}
                  disabled={loading || recording}
                >
                  Track an order
                </button>

                <button
                  type="button"
                  onClick={() => setMessage("How long does shipping take?")}
                  disabled={loading || recording}
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

          {speaking && (
            <div className="audio-status">
              <span className="audio-bars" aria-hidden="true">
                <i />
                <i />
                <i />
                <i />
                <i />
              </span>
              {audioPlaying ? "Assistant is speaking..." : "Preparing audio..."}
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
              disabled={loading || recording || speaking || audioPlaying}
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
                audioPlaying
              }
            >
              {loading ? "Waiting..." : "Send"}
              <span aria-hidden="true">➤</span>
            </button>
          </form>

          <div className="voice-controls">
            {!recording ? (
              <button
                className="voice-button start-button"
                type="button"
                onClick={startRecording}
                disabled={loading || speaking || audioPlaying}
              >
                <span aria-hidden="true">🎙️</span>
                Start voice
              </button>
            ) : (
              <button
                className="voice-button stop-button"
                type="button"
                onClick={stopRecording}
              >
                <span className="stop-square" />
                Stop and send
              </button>
            )}

            <span className="input-hint">
              {recording
                ? "Your transcript stays visible after you stop."
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
