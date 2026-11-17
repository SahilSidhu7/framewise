import type { ReactNode } from "react";

export function Logo({ size = 22, word = true }: { size?: number; word?: boolean }) {
  return (
    <span className="logo">
      <img src="/glyph.svg" width={size} height={size} alt="" />
      {word && <span className="logo-word">Framewise</span>}
    </span>
  );
}

/** Full-window layout for setup and the start screen: brand rail on the left, content on the right. */
export function Shell({ side, children, footer }: { side?: ReactNode; children: ReactNode; footer?: ReactNode }) {
  return (
    <div className="shell">
      <aside className="shell-side">
        <Logo size={26} />
        <div className="shell-side-body">{side}</div>
        <p className="shell-version">Version 0.1 · Windows beta</p>
      </aside>
      <main className="shell-main">
        <div className="shell-content">{children}</div>
        {footer && <div className="shell-footer">{footer}</div>}
      </main>
    </div>
  );
}
