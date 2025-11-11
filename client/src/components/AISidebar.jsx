export default function AISidebar({ history, onSelect }) {
  return (
    <div className="hidden md:flex flex-col w-64 border-r border-gray-300 dark:border-gray-800 bg-white dark:bg-black p-4">
      <h2 className="text-xl font-bold mb-4 text-black dark:text-white">History</h2>
      
      <div className="space-y-2 overflow-y-auto flex-1">
        {history.map((h, i) => (
          <button
            key={i}
            onClick={() => onSelect(i)}
            className="w-full text-left p-3 rounded-lg bg-gray-100 dark:bg-gray-900 hover:bg-gray-200 dark:hover:bg-gray-800 transition"
          >
            <p className="text-sm text-black dark:text-white line-clamp-1">{h.title}</p>
            <p className="text-xs text-gray-500 dark:text-gray-400">{h.date}</p>
          </button>
        ))}
      </div>
    </div>
  );
}
