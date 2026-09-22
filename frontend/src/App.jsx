import { useState } from "react";

function App() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  async function sendMessage() {
    if (!message.trim() || loading) {
      return;
    }

    const userMessage = message;

    // Show user's message immediately
    setMessages((previousMessages) => [
      ...previousMessages,
      {
        role: "user",
        content: userMessage,
      },
    ]);

    setMessage("");
    setLoading(true);

    const response = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: userMessage,
      }),
    });

    const data = await response.json();

    // Add bot response
    setMessages((previousMessages) => [
      ...previousMessages,
      {
        role: "assistant",
        content: data.answer,
      },
    ]);

    setLoading(false);
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

      </div>

    </div>
  );
}

export default App;