import { Link, useLocation } from "react-router-dom";
import { useState } from "react";
import { BookOpen, Bot, Menu, X, User } from "lucide-react";
import ThemeToggle from "./ThemeToggle";
import { motion } from "framer-motion";

export default function Navbar() {
  const location = useLocation();
  const [showMenu, setShowMenu] = useState(false);
  const [showProfileMenu, setShowProfileMenu] = useState(false);

  /** ✅ Only Two Tabs */
  const tabs = [
    { name: "My Courses", path: "/student/courses", icon: BookOpen },
    { name: "Sarvasva AI", path: "/ai", icon: Bot },
  ];

  const activeTab = (path) => location.pathname === path;

  return (
    <nav className="fixed top-0 left-0 right-0 z-40 border-b border-gray-300 dark:border-gray-800 bg-white/95 dark:bg-black/95 backdrop-blur-sm">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          
          {/* ✅ Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-black dark:bg-white text-white dark:text-black font-bold text-xl">
              S
            </div>
            <span className="text-xl font-bold text-black dark:text-white">
              Sarvasva
            </span>
          </Link>

          {/* ✅ Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {tabs.map((tab) => (
              <Link
                key={tab.path}
                to={tab.path}
                className={`flex items-center space-x-1 px-4 py-2 rounded-lg transition-all
                  ${
                    activeTab(tab.path)
                      ? "bg-black dark:bg-white text-white dark:text-black"
                      : "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-black dark:hover:text-white"
                  }`}
              >
                <tab.icon size={18} />
                <span>{tab.name}</span>
              </Link>
            ))}
          </div>

          {/* ✅ Right Controls */}
          <div className="flex items-center space-x-4">
            <ThemeToggle />

            {/* ✅ Profile Button */}
            <div className="relative">
              <button
                onClick={() => setShowProfileMenu(!showProfileMenu)}
                className="flex items-center space-x-2 rounded-lg bg-gray-100 dark:bg-gray-800 px-3 py-2 hover:bg-gray-200 dark:hover:bg-gray-700 text-black dark:text-white"
              >
                <User size={18} />
                <span className="hidden md:block">Guest</span>
              </button>

              {showProfileMenu && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="absolute right-0 mt-2 w-48 rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-2 shadow-xl"
                >
                  <Link
                    to="/profile"
                    onClick={() => setShowProfileMenu(false)}
                    className="flex items-center space-x-2 rounded-lg px-3 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                  >
                    <User size={16} />
                    <span>Profile</span>
                  </Link>
                </motion.div>
              )}
            </div>

            {/* ✅ Mobile Menu Icon */}
            <button
              onClick={() => setShowMenu(!showMenu)}
              className="md:hidden rounded-lg p-2 hover:bg-gray-800"
            >
              {showMenu ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>

        {/* ✅ Mobile Menu */}
        {showMenu && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden border-t border-gray-800 py-4"
          >
            <div className="space-y-2">
              {tabs.map((tab) => (
                <Link
                  key={tab.path}
                  to={tab.path}
                  onClick={() => setShowMenu(false)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg
                    ${
                      activeTab(tab.path)
                        ? "bg-white text-black"
                        : "text-gray-300 hover:bg-gray-800"
                    }`}
                >
                  <tab.icon size={18} />
                  <span>{tab.name}</span>
                </Link>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </nav>
  );
}
