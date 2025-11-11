"use client";

import { useState, useEffect, useRef } from "react";
import { MessageCircle, X, Send } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function FloatingChatbot() {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [chat, setChat] = useState([
    { from: "bot", text: "Hi! 👋 How can I help you today?" },
  ]);

  const chatRef = useRef(null);

  // ✅ Auto scroll to bottom
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [chat]);

  const sendMessage = () => {
    if (!input.trim()) return;

    setChat((prev) => [...prev, { from: "user", text: input }]);

    setTimeout(() => {
      setChat((prev) => [
        ...prev,
        { from: "bot", text: "✅ I received your message!" },
      ]);
    }, 600);

    setInput("");
  };

  return (
    <>
      {/* ✅ Floating Chat Icon */}
      <motion.button
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        whileHover={{ scale: 1.15 }}
        whileTap={{ scale: 0.9 }}
        onClick={() => setOpen(!open)}  // ✅ toggle open/close
        className="fixed bottom-6 right-6 z-50 p-4 rounded-full
                   bg-black dark:bg-white text-white dark:text-black 
                   shadow-xl hover:shadow-2xl transition"
      >
        {open ? <X size={26} /> : <MessageCircle size={28} />}
      </motion.button>

      {/* ✅ Chat Window */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 30, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.9 }}
            transition={{ duration: 0.25 }}
            className="fixed bottom-24 right-6 z-50 w-96 h-[500px]
                       bg-white dark:bg-gray-900 rounded-2xl shadow-2xl
                       border border-gray-200 dark:border-gray-800
                       flex flex-col overflow-hidden"
          >
            {/* ✅ Header */}
            <div className="bg-black dark:bg-white text-white dark:text-black 
                            px-4 py-3 font-semibold flex justify-between">
              <span>Aqua Assistant 🧠</span>
              <button onClick={() => setOpen(false)}>
                <X size={22} />
              </button>
            </div>

            {/* ✅ Chat Body */}
            <div
              ref={chatRef}
              className="flex-1 px-4 py-3 overflow-y-auto
                         bg-gray-100 dark:bg-gray-800 space-y-3"
            >
              {chat.map((msg, i) => (
                <div
                  key={i}
                  className={`max-w-[80%] px-4 py-2 rounded-lg text-sm
                    ${msg.from === "user"
                      ? "ml-auto bg-black dark:bg-white text-white dark:text-black"
                      : "bg-gray-300 dark:bg-gray-700 text-black dark:text-white"
                    }`}
                >
                  {msg.text}
                </div>
              ))}
            </div>

            {/* ✅ Input Area */}
            <div className="border-t border-gray-300 dark:border-gray-700 p-3 flex space-x-2">
              <input
                type="text"
                className="flex-1 px-3 py-2 rounded-lg bg-gray-200 dark:bg-gray-800 
                           text-black dark:text-white outline-none"
                placeholder="Ask something..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              />

              <button
                onClick={sendMessage}
                className="px-4 py-2 rounded-lg bg-black dark:bg-white
                           text-white dark:text-black hover:opacity-80 transition"
              >
                <Send size={18} />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
