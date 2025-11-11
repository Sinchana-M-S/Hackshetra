import { useEffect, useState, useRef } from "react";
import {
  Mic,
  StopCircle,
  Send,
  Menu,
  X,
  MoreVertical,
  Edit,
  Trash2,
  FolderPlus,
  Search,
} from "lucide-react";
import axios from "axios";
import Button from "../components/ui/Button";

const AI_API = "http://127.0.0.1:5000";

export default function Ai() {
  const [history, setHistory] = useState([]);
  const [currentChat, setCurrentChat] = useState(null);
  const [messages, setMessages] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [input, setInput] = useState("");
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunks = useRef([]);

  const chatRef = useRef(null);

  // ✅ Auto scroll down
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  // ✅ Auto-create a chat if none exists
  useEffect(() => {
    if (history.length === 0) {
      newChat();
    } else if (!currentChat) {
      const first = history[0];
      setCurrentChat(first.id);
      setMessages(first.messages);
    }
  }, [history]);

  // ✅ Create new chat session
  const newChat = () => {
    const id = crypto.randomUUID();
    const newChatObj = {
      id,
      name: "New Chat",
      messages: [],
      createdAt: Date.now(),
    };
    setHistory((prev) => [newChatObj, ...prev]);
    setCurrentChat(id);
    setMessages([]);
  };

  // ✅ Generate chat name from first message
  const generateChatName = (text) => {
    return text.split(" ").slice(0, 3).join(" ");
  };

  // ✅ Rename chat
  const renameChat = (id, newName) => {
    setHistory((prev) =>
      prev.map((c) => (c.id === id ? { ...c, name: newName } : c))
    );
  };

  // ✅ Delete chat
  const deleteChat = (id) => {
    const updated = history.filter((c) => c.id !== id);
    setHistory(updated);
    if (updated.length > 0) {
      setCurrentChat(updated[0].id);
      setMessages(updated[0].messages);
    } else {
      newChat();
    }
  };

  // ✅ Send Text Message
  const sendMessage = async () => {
    if (!input.trim()) return;

    let chat = history.find((c) => c.id === currentChat);

    // ✅ Auto-create chat if missing
    if (!chat) {
      newChat();
      chat = history[0];
    }

    const userMessage = { from: "user", text: input };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);

    // ✅ Set chat name based on first message
    if (updatedMessages.length === 1) {
      renameChat(chat.id, generateChatName(input));
    }

    setInput("");

    try {
      const res = await axios.post(`${AI_API}/chat`, {
        message: input,
        session_id: chat.id,
      });

      setMessages((prev) => [
        ...prev,
        { from: "bot", text: res.data.response },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { from: "bot", text: "⚠️ AI Server Error" },
      ]);
    }
  };

  // ✅ File Upload
  const uploadFile = (file) => {
    setMessages((prev) => [
      ...prev,
      { from: "user", text: `📎 Uploaded: ${file.name}` },
    ]);
  };

  const handleFileInput = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "*/*";
    input.onchange = (e) => uploadFile(e.target.files[0]);
    input.click();
  };

  // ✅ Start Voice Recording
  const startRecording = async () => {
    setRecording(true);
    audioChunks.current = [];

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);

    recorder.ondataavailable = (e) => audioChunks.current.push(e.data);
    recorder.onstop = sendVoiceMessage;

    recorder.start();
    mediaRecorderRef.current = recorder;
  };

  // ✅ Stop recording
  const stopRecording = () => {
    setRecording(false);
    mediaRecorderRef.current?.stop();
  };

  // ✅ Convert voice → text → AI response
  const sendVoiceMessage = async () => {
    const audioBlob = new Blob(audioChunks.current, { type: "audio/webm" });
    const formData = new FormData();
    formData.append("audio", audioBlob, "voice.webm");

    try {
      const stt = await axios.post(`${AI_API}/speech-to-text`, formData);
      const text = stt.data.transcription;

      if (text) {
        setInput(text);
      }
    } catch (err) {
      console.log("Voice error");
    }
  };

  return (
    <div className="flex h-screen pt-16 bg-white dark:bg-black text-black dark:text-white">

      {/* ✅ Sidebar */}
      <div
        className={`${
          sidebarOpen ? "w-64" : "w-0"
        } transition-all bg-gray-100 dark:bg-gray-900 border-r border-gray-300 dark:border-gray-700 overflow-hidden`}
      >
        <div className="p-4 border-b border-gray-300 dark:border-gray-700 flex items-center justify-between">
          <h2 className="font-bold text-lg">Chats</h2>
          <Button size="sm" onClick={newChat}>
            New Chat
          </Button>
        </div>

        {/* Search */}
        <div className="p-3 flex items-center gap-2 border-b border-gray-300 dark:border-gray-700">
          <Search size={18} />
          <input
            className="bg-transparent w-full outline-none"
            placeholder="Search chats"
          />
        </div>

        {/* Chat History */}
        <div className="overflow-y-auto h-full">
          {history.map((chat) => (
            <div
              key={chat.id}
              className={`px-4 py-3 border-b border-gray-300 dark:border-gray-800 cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 ${
                currentChat === chat.id ? "bg-gray-300 dark:bg-gray-700" : ""
              }`}
              onClick={() => {
                setCurrentChat(chat.id);
                setMessages(chat.messages);
              }}
            >
              <div className="flex justify-between">
                <span className="truncate">{chat.name}</span>

                {/* 3 dots menu */}
                <div className="relative group">
                  <MoreVertical size={18} />

                  <div className="hidden group-hover:block absolute right-0 mt-2 bg-gray-800 text-white p-2 rounded shadow-lg z-50 w-32">
                    <button
                      className="flex items-center gap-2 w-full text-left p-1 hover:bg-gray-700"
                      onClick={() => {
                        const name = prompt("Rename chat:", chat.name);
                        if (name) renameChat(chat.id, name);
                      }}
                    >
                      <Edit size={14} /> Edit
                    </button>

                    <button
                      className="flex items-center gap-2 w-full text-left p-1 hover:bg-gray-700"
                      onClick={() => deleteChat(chat.id)}
                    >
                      <Trash2 size={14} /> Delete
                    </button>
                  </div>
                </div>
              </div>

              <hr className="border-gray-400/40 mt-2" />
            </div>
          ))}
        </div>
      </div>

      {/* ✅ Toggle Button */}
      <button
        className="absolute left-2 top-20 bg-gray-200 dark:bg-gray-800 px-2 py-1 rounded"
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* ✅ Chat Window */}
      <div className="flex-1 flex flex-col">
        <div
          ref={chatRef}
          className="flex-1 overflow-y-auto p-6 space-y-4"
        >
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`max-w-xl px-4 py-2 rounded-xl ${
                msg.from === "user"
                  ? "ml-auto bg-blue-600 text-white"
                  : "bg-gray-200 dark:bg-gray-800"
              }`}
            >
              {msg.text}
            </div>
          ))}
        </div>

        {/* ✅ Input Section */}
        <div className="p-4 border-t border-gray-300 dark:border-gray-700 flex items-center gap-2">
          {/* File Upload */}
          <button
            onClick={handleFileInput}
            className="p-2 rounded-lg bg-gray-200 dark:bg-gray-800"
          >
            <FolderPlus size={22} />
          </button>

          {/* Voice */}
          {recording ? (
            <button
              onClick={stopRecording}
              className="p-2 bg-red-500 text-white rounded-lg"
            >
              <StopCircle size={24} />
            </button>
          ) : (
            <button
              onClick={startRecording}
              className="p-2 bg-gray-200 dark:bg-gray-800 rounded-lg"
            >
              <Mic size={22} />
            </button>
          )}

          <input
            className="flex-1 px-3 py-2 rounded-lg bg-gray-200 dark:bg-gray-900 outline-none"
            placeholder="Ask something..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />

          <button
            className="p-3 bg-blue-600 text-white rounded-lg"
            onClick={sendMessage}
          >
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}
