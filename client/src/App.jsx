import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { useEffect } from 'react';
import useThemeStore from './store/themeStore';
import Navbar from './components/Navbar';

// Pages
import Home from './pages/Home';
import StudentCourses from './pages/student/StudentCourses';
import StudentCredits from './pages/student/StudentCredits';
import StudentCompleted from './pages/student/StudentCompleted';
import StudentDocuments from './pages/student/StudentDocuments';
import StudentSummarizer from './pages/student/StudentSummarizer';
import StudentChat from './pages/student/StudentChat';
import StudentAssessments from './pages/student/StudentAssessments';
import InstructorUpload from './pages/instructor/InstructorUpload';
import InstructorLiveClasses from './pages/instructor/InstructorLiveClasses';
import InstructorCredits from './pages/instructor/InstructorCredits';
import CourseDetail from './pages/CourseDetail';
import CoursePlayer from './pages/CoursePlayer';
import Profile from './pages/Profile';
import FloatingChatbot from './components/FloatingChatbot';
import Footer from './components/Footer';
import AI from './pages/AI';

function App() {
  const { theme } = useThemeStore();

  // ✅ Theme handling stays same
  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove('dark', 'light');
    root.classList.add(theme);
  }, [theme]);

  useEffect(() => {
    const stored = localStorage.getItem('theme-storage');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        const savedTheme = parsed.state?.theme || 'dark';
        const root = document.documentElement;
        root.classList.remove('dark', 'light');
        root.classList.add(savedTheme);
      } catch (e) {
        document.documentElement.classList.add('dark');
      }
    } else {
      document.documentElement.classList.add('dark');
    }
  }, []);

  return (
    <Router>
      <div className="min-h-screen bg-white dark:bg-black text-black dark:text-white">
        <Navbar />

        <main>
          <Routes>
            <Route path="/" element={<Home />} />

            {/* ✅ No login route */}
            {/* <Route path="/login" element={<Login />} /> */}

            {/* PUBLIC course pages */}
            <Route path="/course/:id" element={<CourseDetail />} />
            <Route path="/course/:id/play" element={<CoursePlayer />} />
            <Route path="/ai" element={<AI />} />


            {/* ✅ Student Routes – all public */}
            <Route path="/student/courses" element={<StudentCourses />} />
            <Route path="/student/credits" element={<StudentCredits />} />
            <Route path="/student/completed" element={<StudentCompleted />} />
            <Route path="/student/documents" element={<StudentDocuments />} />
            <Route path="/student/summarizer" element={<StudentSummarizer />} />
            <Route path="/student/chat" element={<StudentChat />} />
            <Route path="/student/assessments" element={<StudentAssessments />} />

            {/* ✅ Instructor Routes – all public */}
            <Route path="/instructor/upload" element={<InstructorUpload />} />
            <Route path="/instructor/live-classes" element={<InstructorLiveClasses />} />
            <Route path="/instructor/credits" element={<InstructorCredits />} />

            {/* ✅ Profile – public & stored locally */}
            <Route path="/profile" element={<Profile />} />
          </Routes>
        </main>

        {/* ✅ Floating chatbot always visible */}
         {location.pathname === "/" && <FloatingChatbot />}

        <Toaster
          position="top-right"
          toastOptions={{
            style: {
              background: 'var(--toast-bg, #1f1f1f)',
              color: 'var(--toast-color, #fff)',
              border: '1px solid var(--toast-border, #333)',
            },
            className: 'dark:bg-gray-900 dark:text-white bg-white dark:text-white',
          }}
        />
      </div>
    </Router>
  );
}

export default App;
