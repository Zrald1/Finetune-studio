import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';

// Suppress the webview's default context menu so the app reads as a native
// desktop tool rather than a web page. Editable fields are exempted: the native
// menu is the only right-click paste path on some setups, and losing it in a
// text box is a usability regression rather than a polish win.
function isEditable(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || !el.tagName) return false;
  const tag = el.tagName.toLowerCase();
  return (
    tag === "input" ||
    tag === "textarea" ||
    tag === "select" ||
    el.isContentEditable === true
  );
}

document.addEventListener("contextmenu", (e) => {
  if (!isEditable(e.target)) e.preventDefault();
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
