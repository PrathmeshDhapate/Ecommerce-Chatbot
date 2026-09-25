import { useRef, useState } from "react";
import "./App.css"

function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);

  const [threadId] = useState(() => crypto.randomUUID());

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });

    const recorder = new MediaRecorder(stream, {
      mimeType: "audio/webm;codecs=opus",
    });

    audioChunksRef.current = [];

    recorder.ondataavailable = (event) => {
      audioChunksRef.current.push(event.data);
    };

    mediaRecorderRef.current = recorder;

    recorder.start();

    setRecording(true);
  }

  function stopRecording() {
    const recorder = mediaRecorderRef.current;

    recorder.onstop = async () => {
      const audioBlob = new Blob(audioChunksRef.current, {
        type: "audio/webm;codecs=opus",
      });

      const formData = new FormData();

      formData.append("audio", audioBlob, "recording.webm");
      formData.append("thread_id", threadId);

      try {
        const response = await fetch("http://127.0.0.1:8000/voice", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error("Voice request failed");
        }

        const data = await response.json();

        // Convert Gemini's Base64 audio into binary bytes
        const audioBytes = Uint8Array.from(
          atob(data.audio_base64),
          (character) => character.charCodeAt(0),
        );

        // Create a WAV audio blob
        const audioBlob = new Blob([audioBytes], { type: "audio/wav" });

        // Create a temporary URL for the audio
        const audioUrl = URL.createObjectURL(audioBlob);

        // Play Gemini-generated audio
        const audio = new Audio(audioUrl);

        await audio.play();

        // Clean up the temporary URL after playback starts
        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
        };

        setMessages((previousMessages) => [
          ...previousMessages,
          {
            role: "user",
            content: data.transcription,
          },
          {
            role: "assistant",
            content: data.answer,
          },
        ]);
      } catch (error) {
        console.error("Voice request failed:", error);

        setMessages((previousMessages) => [
          ...previousMessages,
          {
            role: "assistant",
            content: "Sorry, something went wrong with voice input.",
          },
        ]);
      }
    };

    recorder.stop();

    recorder.stream.getTracks().forEach((track) => {
      track.stop();
    });

    setRecording(false);
  }

  async function sendMessage() {
    if (!message.trim() || loading) {
      return;
    }

    const userMessage = message;

    setMessages((previousMessages) => [
      ...previousMessages,
      {
        role: "user",
        content: userMessage,
      },
    ]);

    setMessage("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
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

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: "assistant",
          content: data.answer,
        },
      ]);
    } catch (error) {
      console.error("Chat request failed:", error);

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: "assistant",
          content: "Sorry, something went wrong. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

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

        {loading && <div className="message assistant">Thinking...</div>}
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
        />

        <button onClick={sendMessage} disabled={loading}>
          {loading ? "Sending..." : "Send"}
        </button>

        <button onClick={startRecording} disabled={recording}>
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
