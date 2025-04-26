import { useState, createContext, useContext, useEffect } from "react";

const ThemeContext = createContext<{
  toggleTheme: () => void;
  theme: string;
} | null>(null);

export const useTheme = () => {
  const theme = useContext(ThemeContext);
  if (!theme) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return theme;
};

export const ThemeProvider = ({ children }: { children: React.ReactNode }) => {
  const themes = ["Pink", "Blue", "Purple", "Orange"];
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("theme") || "Orange";
  });

  const toggleTheme = () => {
    const newTheme = themes[(themes.indexOf(theme) + 1) % themes.length];
    localStorage.setItem("theme", newTheme);
  };

  useEffect(() => {
    const root = document.documentElement;
    switch (theme) {
      case "Pink":
        root.style.setProperty("--color-background", "#250821");
        root.style.setProperty("--color-ascent", "oklch(0.656 0.241 354.308)");
        root.style.setProperty("--color-aslight", "oklch(0.718 0.202 349.761)");
        root.style.setProperty("--color-asdark", "oklch(0.525 0.223 3.958)");
        root.style.setProperty("--color-gstart", "rgb(254, 0, 242)");
        root.style.setProperty("--color-gstop", "rgb(251, 2, 128)");
        break;
      case "Blue":
        root.style.setProperty("--color-background", "#14315c");
        root.style.setProperty("--color-ascent", "oklch(0.623 0.214 259.815)");
        root.style.setProperty("--color-aslight", "oklch(0.707 0.165 254.624)");
        root.style.setProperty("--color-asdark", "oklch(0.546 0.245 262.881)");
        root.style.setProperty("--color-gstart", "rgb(31, 38, 145)");
        root.style.setProperty("--color-gstop", "rgb(71, 128, 227)");
        break;
      case "Purple":
        root.style.setProperty("--color-background", "#271630");
        root.style.setProperty("--color-ascent", "oklch(0.627 0.265 303.9)");
        root.style.setProperty("--color-aslight", "oklch(0.714 0.203 305.504)");
        root.style.setProperty("--color-asdark", "oklch(0.558 0.288 302.321)");
        root.style.setProperty("--color-gstart", "rgb(48, 5, 70)");
        root.style.setProperty("--color-gstop", "rgb(152, 108, 221)");
        break;
      case "Orange":
        root.style.setProperty("--color-background", "#3d2616");
        root.style.setProperty("--color-ascent", "oklch(0.705 0.213 47.604)");
        root.style.setProperty("--color-aslight", "oklch(0.75 0.183 55.934)");
        root.style.setProperty("--color-asdark", "oklch(0.646 0.222 41.116)");
        root.style.setProperty("--color-gstart", "rgb(253, 200, 48)");
        root.style.setProperty("--color-gstop", "rgb(243, 115, 53)");
        break;
      default:
        root.style.setProperty("--color-background", "#250821");
        root.style.setProperty("--color-ascent", "oklch(0.656 0.241 354.308)");
        root.style.setProperty("--color-aslight", "oklch(0.718 0.202 349.761)");
        root.style.setProperty("--color-asdark", "oklch(0.525 0.223 3.958)");
        root.style.setProperty("--color-gstart", "rgb(254, 0, 242)");
        root.style.setProperty("--color-gstart", "rgb(251, 2, 128)");
    }
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ toggleTheme, theme }}>
      {children}
    </ThemeContext.Provider>
  );
};
