import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { MachineProfile } from "../lib/machine";
import type { NavigationPage } from "../lib/navigation";
import { goToEntry, isEditable, SEARCH_RESULT_LIMIT, searchEntries, searchMatches } from "../lib/navigationSearch";
import { Icon, ICON_SIZE } from "./Icon";
import { ModalShell } from "./ModalShell";
import { Button } from "./ui";

/**
 * The top bar's Search (Ctrl+K or Cmd+K): find a page, a tab or a Settings
 * section by name and go there. It searches names only (lib/navigationSearch).
 */
export function CommandSearch({
  allowed,
  machine,
  memoryAllowed = false,
}: {
  allowed: (page: NavigationPage) => boolean;
  machine: MachineProfile;
  /** Memory is administrator-only and is not a NavigationPage, so it is gated apart. */
  memoryAllowed?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const entries = useMemo(() => searchEntries(machine, allowed, memoryAllowed), [machine, allowed, memoryAllowed]);
  // An empty query lists every page this role may open; a typed one, the best few.
  const matches = useMemo(() => {
    const found = searchMatches(entries, query);
    return query.trim() ? found.slice(0, SEARCH_RESULT_LIMIT) : found;
  }, [entries, query]);

  useEffect(() => {
    const shortcut = (event: KeyboardEvent) => {
      // Inside a text field Ctrl+K belongs to the field (a terminal, an editor), not to Search.
      if (isEditable(event.target)) return;
      // An open dialog owns the keyboard. Search lives in the header, which is
      // inert beneath it, so opening it there would stack a second dialog that
      // Escape could not reach first.
      if (document.querySelector("[aria-modal='true']")) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen(true);
      }
    };
    window.addEventListener("keydown", shortcut);
    return () => window.removeEventListener("keydown", shortcut);
  }, []);

  useEffect(() => { setActive(0); }, [query]);

  // The keyboard's row is always on screen: ArrowDown past the list's visible
  // end scrolls it, so Enter never opens a row the owner could not see.
  const activeId = open && matches[active] ? `command-search-${matches[active].id}` : "";
  useEffect(() => {
    if (activeId) document.getElementById(activeId)?.scrollIntoView?.({ block: "nearest" });
  }, [activeId]);

  const close = () => { setOpen(false); setQuery(""); };
  const choose = (index: number) => {
    const entry = matches[index];
    if (!entry) return;
    close();
    goToEntry(entry);
  };

  return (
    <>
      <Button aria-keyshortcuts="Control+K Meta+K" className="topbar__search" onClick={() => setOpen(true)}>
        <Icon name="search" size={ICON_SIZE.inline} />
        <span>Search</span>
        <kbd>Ctrl K</kbd>
      </Button>
      {/*
        * On the document body, not in the header: the top bar's backdrop-filter
        * makes it the containing block of every position: fixed descendant, so
        * a dialog rendered inside it is a 64 px strip at the top of the page
        * (the owner's screenshot), whatever its own CSS says.
        */}
      {open && createPortal(
        <ModalShell backdropClassName="command-search__backdrop" className="command-search" initialFocusRef={input} labelledBy="command-search-label" onClose={close}>
          <label className="command-search__field" htmlFor="command-search-input" id="command-search-label">
            <Icon name="search" size={16} />
            <span className="sr-only">Search pages, tabs and settings</span>
            <input
              aria-activedescendant={matches[active] ? `command-search-${matches[active].id}` : undefined}
              aria-autocomplete="list"
              aria-controls="command-search-results"
              aria-expanded="true"
              autoComplete="off"
              className="command-search__input"
              id="command-search-input"
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "ArrowDown") { event.preventDefault(); setActive((index) => Math.min(index + 1, matches.length - 1)); }
                if (event.key === "ArrowUp") { event.preventDefault(); setActive((index) => Math.max(index - 1, 0)); }
                if (event.key === "Enter") { event.preventDefault(); choose(active); }
              }}
              placeholder="Search pages, tabs and settings"
              ref={input}
              role="combobox"
              type="search"
              value={query}
            />
          </label>
          <ul aria-label="Results" className="command-search__results" id="command-search-results" role="listbox">
            {matches.map((entry, index) => (
              <li
                aria-selected={index === active}
                className={index === active ? "command-search__result is-active" : "command-search__result"}
                id={`command-search-${entry.id}`}
                key={entry.id}
                onClick={() => choose(index)}
                onMouseMove={() => setActive(index)}
                role="option"
              >
                {/* The space keeps the option's name "Memory Page", not "MemoryPage". */}
                <span className="command-search__label">{entry.label}</span>{" "}
                <small className="command-search__context">{entry.context}</small>
              </li>
            ))}
          </ul>
          {!matches.length && <p className="command-search__empty">Nothing is called that. Search finds pages, tabs and settings by name.</p>}
        </ModalShell>,
        document.body,
      )}
    </>
  );
}
