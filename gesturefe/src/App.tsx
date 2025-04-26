import "./App.css";
import { About } from "./components/about";
import Background from "./components/background";
import { Dashboard } from "./components/dashboard";
import { Home } from "./components/home";
import { Navbar } from "./components/navbar";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Ownership } from "./components/ownership";
import { ThemeProvider } from "./components/themeProvider";
import DrawingBoard from "./components/drawingBoard";

function App() {
  return (
    <>
      <ThemeProvider>
        <BrowserRouter>
          <Navbar />
          <Background />
          <Ownership />
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/dashboard" element={<Dashboard />} />
            path="/signup"
            <Route path="*" element={<Navigate to="/" />} />
            <Route path="/about" element={<About />} />
            <Route path="/draw" element={<DrawingBoard />} />
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </>
  );
}

export default App;
