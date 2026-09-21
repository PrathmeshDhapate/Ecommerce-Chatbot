import { useState } from "react";

function App() {

  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState("");

  async function sendMessage() {

    const response = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message,
      }),
    });

    const data = await response.json();

    setAnswer(data.answer);
  }

  return (
    <div>

      <h1>Commerce Chatbot</h1>

      <input
        value={message}
        onChange={(event) => setMessage(event.target.value)}
        placeholder="Ask something..."
      />

      <button onClick={sendMessage}>
        Send
      </button>

      <p>{answer}</p>

    </div>
  );
}

export default App;