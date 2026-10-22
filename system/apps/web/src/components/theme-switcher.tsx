"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useSyncExternalStore } from "react";

const subscribe = () => () => {};
const clientSnapshot = () => true;
const serverSnapshot = () => false;
const modes = [
  { value: "light", label: "白天模式", icon: Sun },
  { value: "dark", label: "夜间模式", icon: Moon },
  { value: "system", label: "跟随系统", icon: Monitor },
];

export function ThemeSwitcher() {
  const { theme, setTheme } = useTheme();
  // The stored preference is only available in the browser; keep SSR and hydration identical.
  const mounted = useSyncExternalStore(subscribe, clientSnapshot, serverSnapshot);
  return (
    <div className="theme-switcher" role="group" aria-label="外观模式">
      {modes.map(({ value, label, icon: Icon }) => (
        <button key={value} type="button" title={label} aria-label={label} aria-pressed={mounted && theme === value} disabled={!mounted} onClick={() => setTheme(value)}>
          <Icon size={16} strokeWidth={1.7} />
        </button>
      ))}
    </div>
  );
}
