export default function AIMessage({ from, text }) {
  const isUser = from === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[70%] rounded-xl px-4 py-2 text-sm shadow-md 
          ${isUser 
            ? "bg-blue-600 text-white rounded-br-none" 
            : "bg-gray-200 dark:bg-gray-800 text-black dark:text-white rounded-bl-none"
          }`}
      >
        {text}
      </div>
    </div>
  );
}
