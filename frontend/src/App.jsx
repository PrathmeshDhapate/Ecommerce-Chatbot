import { useRef, useState } from "react";

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

      formData.append(
        "audio",
        audioBlob,
        "recording.webm"
      );

      try {
        const response = await fetch(
          "http://127.0.0.1:8000/voice",
          {
            method: "POST",
            body: formData,
          }
        );

        if (!response.ok) {
          throw new Error("Voice request failed");
        }

        const data = await response.json();

        console.log(data);
      } catch (error) {
        console.error("Voice request failed:", error);
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
      const response = await fetch(
        "http://127.0.0.1:8000/chat",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: userMessage,
            thread_id: threadId,
          }),
        }
      );

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

      <h1>Commerce Chatbot</h1>

      <div className="messages">

        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.role}`}
          >
            {msg.content}
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            Thinking...
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
          placeholder="Ask something..."
        />

        <button
          onClick={sendMessage}
          disabled={loading}
        >
          {loading ? "Sending..." : "Send"}
        </button>

        <button
          onClick={startRecording}
          disabled={recording}
        >
          Start Recording
        </button>

        <button
          onClick={stopRecording}
          disabled={!recording}
        >
          Stop Recording
        </button>

      </div>

    </div>
  );
}

export default App;