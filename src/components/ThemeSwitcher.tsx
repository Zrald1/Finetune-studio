import { useEffect, useState } from "react";
import { Palette } from "lucide-react";

const THEMES = [
  { id: "ember", label: "Ember" },
  { id: "graphite", label: "Graphite" },
  { id: "azure", label: "Azure" },
  { id: "sage", label: "Sage" },
  { id: "noir", label: "Noir" },
] as const;

type ThemeId = (typeof THEMES)[number]["id"];

const isThemeId = (value: string | null): value is ThemeId =>
  THEMES.some((theme) => theme.id === value);

export default function ThemeSwitcher() {
  const [theme, setTheme] = useState<ThemeId>(() => {
    if (typeof window === "undefined") return "ember";
    const saved = window.localStorage.getItem("fine-tune-theme");
    return isThemeId(saved) ? saved : "ember";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("fine-tune-theme", theme);
  }, [theme]);

  return (
    <label className="hidden sm:flex items-center gap-2 rounded-md border theme-surface-soft px-3 py-1.5 text-[10px] uppercase tracking-widest font-mono font-bold theme-muted">
      <Palette className="w-3.5 h-3.5 theme-accent" />
      <select
        value={theme}
        onChange={(e) => setTheme(e.target.value as ThemeId)}
        className="bg-transparent border-0 outline-none text-[10px] uppercase tracking-widest font-mono font-bold theme-muted cursor-pointer"
        aria-label="Theme"
      >
        {THEMES.map((item) => (
          <option key={item.id} value={item.id} className="bg-black text-white">
            {item.label}
          </option>
        ))}
      </select>
    </label>
  );
}
