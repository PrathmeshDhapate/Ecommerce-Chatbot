import { useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000";

function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState("");

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
  const voiceTimingRef = useRef({
    startedAt: 0,
    firstChunkReceived: false,
    firstPlaybackStarted: false,
  });

  const [audioPlaying, setAudioPlaying] = useState(false);
  const [speaking, setSpeaking] = useState(false);

  const nextPlayTimeRef = useRef(0);
  const audioQueueRef = useRef(Promise.resolve());

  const transcriptRef = useRef("");

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
        console.warn("Failed to disconnect processorRef.current");
      }

      processorRef.current = null;
    }

    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch {
        console.warn("Failed to disconnect sourceRef.current");
      }

      sourceRef.current = null;
    }

    if (gainRef.current) {
      try {
        gainRef.current.disconnect();
      } catch {
        console.warn("Failed to disconnect gainRef.current");
      }

      gainRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });

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

  async function playAudioChunk(arrayBuffer) {
    let audioContext = playbackAudioContextRef.current;

    if (!audioContext || audioContext.state === "closed") {
      audioContext = new AudioContext();
      playbackAudioContextRef.current = audioContext;
    }

    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    // Convert the incoming 16-bit PCM data to browser audio.
    const pcm = new Int16Array(arrayBuffer);

    if (pcm.length === 0) return;

    const audioBuffer = audioContext.createBuffer(1, pcm.length, 24000);

    const channel = audioBuffer.getChannelData(0);

    for (let i = 0; i < pcm.length; i++) {
      channel[i] = pcm[i] / (pcm[i] < 0 ? 32768 : 32767);
    }

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    // Schedule chunks one after another.
    const startAt = Math.max(
      audioContext.currentTime + 0.02,
      nextPlayTimeRef.current,
    );

    activeAudioSourcesRef.current.add(source);

    source.onended = () => {
      activeAudioSourcesRef.current.delete(source);

      // Hide the indicator only after all audio has played.
      if (ttsFinishedRef.current && activeAudioSourcesRef.current.size === 0) {
        setSpeaking(false);
        setAudioPlaying(false);
      }
    };

    source.start(startAt);

const timing = voiceTimingRef.current;

if (!timing.firstPlaybackStarted) {
  timing.firstPlaybackStarted = true;

  const delayMs = Math.max(
    0,
    (startAt - audioContext.currentTime) * 1000
  );

  console.log(
    `First chunk scheduled for playback: ${
      ((performance.now() - timing.startedAt + delayMs) / 1000).toFixed(3)
    } seconds after tts_start`
  );
}

    nextPlayTimeRef.current = startAt + audioBuffer.duration;

    setAudioPlaying(true);
  }

  // ==========================================================
  // START STT STREAMING
  // ==========================================================

  async function startRecording() {
    if (recording || loading || speaking || audioPlaying) return;

    try {
      // Prepare the playback context during the user's button click.
      if (
        !playbackAudioContextRef.current ||
        playbackAudioContextRef.current.state === "closed"
      ) {
        playbackAudioContextRef.current = new AudioContext();
      }

      await playbackAudioContextRef.current.resume();

      setLiveTranscript("");
      transcriptRef.current = "";

      // Ask permission to use the microphone.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      streamRef.current = stream;

      // Connect to our FastAPI WebSocket endpoint.
      const socket = new WebSocket(
        `${WS_URL}/voice-stream?thread_id=${encodeURIComponent(threadId)}`,
      );
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;

      let resolveReady;
      let rejectReady;

      const readyPromise = new Promise((resolve, reject) => {
        resolveReady = resolve;
        rejectReady = reject;
      });

      socket.onmessage = (event) => {
        // Binary audio chunk received from backend.
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

        if (data.type === "tts_start") {
  voiceTimingRef.current = {
    startedAt: performance.now(),
    firstChunkReceived: false,
    firstPlaybackStarted: false,
  };

  console.log("TTS generation started in frontend");

  setSpeaking(true);
  setAudioPlaying(false);
  nextPlayTimeRef.current = 0;
  ttsFinishedRef.current = false;

  return;
}

        if (data.type === "tts_end") {
          console.log(
    `Backend finished sending TTS: ${
      ((performance.now() - voiceTimingRef.current.startedAt) / 1000).toFixed(3)
    } seconds after tts_start`
  );
          // Wait until all received audio chunks have been scheduled.
          audioQueueRef.current = audioQueueRef.current.then(() => {
            ttsFinishedRef.current = true;

            // There may be no active sources if the audio was empty.
            if (activeAudioSourcesRef.current.size === 0) {
              setSpeaking(false);
              setAudioPlaying(false);
            }
          });

          return;
        }

        if (data.type === "tts_error") {
          setSpeaking(false);
          console.error("TTS error:", data.message);
          return;
        }
        // Backend and Gemini Live are ready.
        if (data.type === "ready") {
          resolveReady();
          return;
        }

        // Live partial transcription.
        if (data.type === "interim") {
          setLiveTranscript(data.text);
          return;
        }

        // Final transcription.
        if (data.type === "final") {
          transcriptRef.current = data.text;
          setLiveTranscript(data.text);
          return;
        }

        // LangGraph response (text only in this phase).
        if (data.type === "answer") {
          transcriptRef.current = data.transcription;

          setMessages((previous) => [
            ...previous,
            {
              role: "user",
              content: data.transcription,
            },
            {
              role: "assistant",
              content: data.answer,
            },
          ]);

          setLiveTranscript("");
          setLoading(false);
          return;
        }

        if (data.type === "error") {
          setMessages((previous) => [
            ...previous,
            {
              role: "assistant",
              content: data.message,
            },
          ]);

          setLoading(false);
          setRecording(false);
          setLiveTranscript("");
        }
      };

      socket.onerror = () => {
        rejectReady(new Error("WebSocket connection failed."));
      };

      socket.onclose = () => {
        setRecording(false);
        setLoading(false);
      };

      // Wait until the WebSocket connection opens.
      await new Promise((resolve, reject) => {
        socket.addEventListener("open", resolve, { once: true });
        socket.addEventListener(
          "error",
          () => reject(new Error("Could not connect to FastAPI.")),
          { once: true },
        );
      });

      // Wait until Gemini Live is connected on the backend.
      await readyPromise;

      // Set up browser audio processing.
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      await audioContext.resume();

      const source = audioContext.createMediaStreamSource(stream);

      sourceRef.current = source;

      // Process microphone audio in small chunks.
      const processor = audioContext.createScriptProcessor(4096, 1, 1);

      processorRef.current = processor;

      // Keep the processor running without playing microphone audio.
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

      setRecording(true);
    } catch (error) {
      console.error("Could not start voice streaming:", error);

      await cleanupMicrophone();

      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }

      setRecording(false);

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content:
            "Could not start voice input. Please check microphone permission and backend connection.",
        },
      ]);
    }
  }

  // ==========================================================
  // STOP STT STREAMING
  // ==========================================================

  async function stopRecording() {
    if (!recording) return;

    setRecording(false);
    setLoading(true);

    await cleanupMicrophone();

    const socket = socketRef.current;

    if (socket && socket.readyState === WebSocket.OPEN) {
      // Tell FastAPI that no more audio is coming.
      socket.send(JSON.stringify({ type: "stop" }));
    } else {
      setLoading(false);
    }
  }

  // ==========================================================
  // NORMAL TEXT CHAT - UNCHANGED
  // ==========================================================

  async function sendMessage() {
    if (!message.trim() || loading || recording || speaking || audioPlaying)
      return;

    const userMessage = message.trim();

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: userMessage,
      },
    ]);

    setMessage("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
          thread_id: threadId,
        }),
      });

      if (!response.ok) {
        throw new Error("Server error");
      }

      const data = await response.json();

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.answer,
        },
      ]);
    } catch (error) {
      console.error("Chat request failed:", error);

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: "Sorry, something went wrong. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  // ==========================================================
  // UI
  // ==========================================================

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Commerce Assistant</h1>
        <span>Online</span>
      </div>

      <div className="messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}

        {recording && (
          <div className="message user">{liveTranscript || "Listening..."}</div>
        )}

        {loading && <div className="message assistant">Thinking...</div>}

        {/* ADD THIS NEW SECTION */}
        {speaking && (
          <div className="message assistant">
            {audioPlaying ? "🔊 Speaking..." : "⏳ Preparing audio..."}
          </div>
        )}
      </div>

      <div className="input-area">
        <input
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              sendMessage();
            }
          }}
          placeholder="Ask about orders, returns, shipping..."
          disabled={recording || loading}
        />

        <button onClick={sendMessage} disabled={loading || recording}>
          {loading ? "Sending..." : "Send"}
        </button>

        <button onClick={startRecording} disabled={recording || loading}>
          🎙 Start
        </button>

        <button onClick={stopRecording} disabled={!recording}>
          ⏹ Stop
        </button>
      </div>
    </div>
  );
}

export default App;
