import { type CSSProperties, type ReactNode } from "react";

/*
 * Models answer in Markdown whether or not the surface renders it. AI Chat
 * showed the source: literal `**Docker in plain language**`, `## Week 1`, and
 * pipe tables printed as rows of `|---|---|`. A table answer was a wall of
 * pipes.
 *
 * This renderer builds React elements directly. Nothing is ever passed to
 * dangerouslySetInnerHTML and no HTML in the model's output is interpreted, so
 * a document that contains `<img onerror=...>` renders as that text and cannot
 * execute. It covers what models actually emit - headings, lists, tables,
 * fenced code, emphasis, inline code, links - and leaves anything else as
 * plain text rather than guessing.
 *
 * Links are deliberately inert. The text of a retrieved document is untrusted,
 * so a URL inside an answer is shown, never made clickable.
 */

type Block =
  | { kind: "heading"; level: number; text: string }
  | { kind: "paragraph"; text: string }
  | { kind: "code"; text: string; language: string }
  | { kind: "list"; ordered: boolean; items: string[]; start: number }
  | { kind: "quote"; text: string }
  | { kind: "table"; header: string[]; rows: string[][] }
  | { kind: "rule" };

const HEADING = /^(#{1,6})\s+(.*)$/;
const UNORDERED = /^\s{0,3}[-*+]\s+(.*)$/;
const ORDERED = /^\s{0,3}(\d{1,3})[.)]\s+(.*)$/;
const QUOTE = /^\s{0,3}>\s?(.*)$/;
const FENCE = /^\s{0,3}(?:```|~~~)\s*([A-Za-z0-9+#_-]*)\s*$/;
const RULE = /^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/;
const TABLE_DIVIDER = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$/;

function tableCells(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

export function parseMarkdown(source: string): Block[] {
  const lines = String(source ?? "").replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let paragraph: string[] = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ kind: "paragraph", text: paragraph.join("\n") });
      paragraph = [];
    }
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const fence = FENCE.exec(line);
    if (fence) {
      flushParagraph();
      const body: string[] = [];
      index += 1;
      while (index < lines.length && !FENCE.test(lines[index])) {
        body.push(lines[index]);
        index += 1;
      }
      blocks.push({ kind: "code", text: body.join("\n"), language: fence[1] ?? "" });
      continue;
    }
    if (!line.trim()) {
      flushParagraph();
      continue;
    }
    if (RULE.test(line)) {
      flushParagraph();
      blocks.push({ kind: "rule" });
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading) {
      flushParagraph();
      blocks.push({ kind: "heading", level: heading[1].length, text: heading[2] });
      continue;
    }
    // A table needs its divider row to be a table at all; without it the same
    // pipes are ordinary text and stay ordinary text.
    if (line.includes("|") && index + 1 < lines.length && TABLE_DIVIDER.test(lines[index + 1])) {
      flushParagraph();
      const header = tableCells(line);
      const rows: string[][] = [];
      index += 2;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(tableCells(lines[index]));
        index += 1;
      }
      index -= 1;
      blocks.push({ kind: "table", header, rows });
      continue;
    }
    if (UNORDERED.test(line) || ORDERED.test(line)) {
      flushParagraph();
      const ordered = !UNORDERED.test(line);
      /*
       * A model asked for twenty numbered bridges writes them with a blank
       * line between each one, and this used to end the list at the first of
       * those blanks. The reader was shown twenty separate one-item lists,
       * every one of them numbered "1." - and the same fault renumbered the
       * four steps of a longer answer to "1." four times. A blank line only
       * ends the list when what follows it is not another item of the same
       * kind, which is what a loose list means in Markdown.
       */
      const continues = (from: number): number => {
        let ahead = from;
        while (ahead < lines.length && !lines[ahead].trim()) ahead += 1;
        if (ahead >= lines.length) return -1;
        const next = lines[ahead];
        const same = ordered ? ORDERED.test(next) && !UNORDERED.test(next) : UNORDERED.test(next);
        return same ? ahead : -1;
      };
      // An explicit first number is the author's, not the renderer's: an answer
      // that resumes at "7." has to keep saying 7.
      const start = ordered ? Number(ORDERED.exec(line)![1]) : 1;
      const items: string[] = [];
      while (index < lines.length) {
        if (!lines[index].trim()) {
          const resumes = continues(index + 1);
          if (resumes < 0) break;
          index = resumes;
          continue;
        }
        const unordered = UNORDERED.exec(lines[index]);
        const numbered = ORDERED.exec(lines[index]);
        if (ordered && numbered && !unordered) items.push(numbered[2]);
        else if (!ordered && unordered) items.push(unordered[1]);
        else if (items.length && /^\s{2,}\S/.test(lines[index])) {
          items[items.length - 1] += `\n${lines[index].trim()}`;
        } else break;
        index += 1;
      }
      index -= 1;
      blocks.push({ kind: "list", ordered, items, start });
      continue;
    }
    const quote = QUOTE.exec(line);
    if (quote) {
      flushParagraph();
      const body = [quote[1]];
      index += 1;
      while (index < lines.length && QUOTE.test(lines[index])) {
        body.push(QUOTE.exec(lines[index])![1]);
        index += 1;
      }
      index -= 1;
      blocks.push({ kind: "quote", text: body.join("\n") });
      continue;
    }
    paragraph.push(line);
  }
  flushParagraph();
  return blocks;
}

/*
 * The link destination allows one level of balanced brackets. `[^)\s]+` ended
 * the URL at its first `)`, which for the shape a model quotes most often -
 * `https://en.wikipedia.org/wiki/Docker_(software)` - falls inside the URL. The
 * reader was shown `..._(software` as the destination with the missing bracket
 * left behind as loose text, and since the destination is displayed precisely
 * so it can be checked, a wrong one is worse than none. The two alternatives
 * differ on their first character, so there is nothing here to backtrack over.
 */
const INLINE = /(`[^`]+`)|(\*\*[^*]+\*\*)|(__[^_]+__)|(\*[^*\n]+\*)|(\[[^\]]+\]\((?:[^\s()]|\([^\s()]*\))+\))/;

/** Render emphasis, inline code and link text. Never produces raw HTML. */
export function renderInline(text: string, keyPrefix = ""): ReactNode[] {
  const nodes: ReactNode[] = [];
  let rest = String(text ?? "");
  let key = 0;
  while (rest) {
    const match = INLINE.exec(rest);
    if (!match || match.index === undefined) {
      nodes.push(rest);
      break;
    }
    if (match.index > 0) nodes.push(rest.slice(0, match.index));
    const token = match[0];
    const id = `${keyPrefix}i${key}`;
    key += 1;
    if (token.startsWith("`")) {
      nodes.push(<code key={id}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith("**") || token.startsWith("__")) {
      nodes.push(<strong key={id}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("*")) {
      nodes.push(<em key={id}>{token.slice(1, -1)}</em>);
    } else {
      // Link text only. The destination is untrusted document text, so it is
      // shown beside the label rather than made clickable.
      const label = token.slice(1, token.indexOf("]("));
      const href = token.slice(token.indexOf("](") + 2, -1);
      nodes.push(
        <span key={id}>
          {label} <small>({href})</small>
        </span>,
      );
    }
    rest = rest.slice(match.index + token.length);
  }
  return nodes;
}

/*
 * These live here rather than in ai-chat.css only because this change may not
 * touch the stylesheet. Every value is a design token, so the result still
 * follows the theme in both light and dark; they should move into
 * `.ai-chat-markdown` rules when the stylesheet is next opened.
 */
const style: Record<string, CSSProperties> = {
  root: { display: "grid", gap: "var(--space-3)", color: "var(--text-secondary)", lineHeight: 1.72 },
  paragraph: { margin: 0, overflowWrap: "anywhere" },
  heading: { margin: 0, color: "var(--text-primary)" },
  list: { margin: 0, paddingInlineStart: "var(--space-4)", display: "grid", gap: "var(--space-1)" },
  quote: {
    margin: 0,
    padding: "var(--space-2) var(--space-3)",
    borderInlineStart: "2px solid var(--border-strong)",
    color: "var(--text-muted)",
  },
  scroll: { overflowX: "auto", maxWidth: "100%" },
  table: { borderCollapse: "collapse", width: "100%", fontSize: "var(--font-size-body-sm)" },
  cell: { padding: "var(--space-2)", border: "1px solid var(--border-subtle)", textAlign: "start" },
  head: {
    padding: "var(--space-2)",
    border: "1px solid var(--border-subtle)",
    textAlign: "start",
    color: "var(--text-primary)",
    background: "var(--bg-hover)",
  },
  pre: {
    margin: 0,
    padding: "var(--space-3)",
    overflowX: "auto",
    background: "var(--bg-surface-raised)",
    border: "1px solid var(--border-subtle)",
    fontFamily: "var(--font-family-data)",
    fontSize: "var(--font-size-body-sm)",
  },
  rule: { margin: 0, border: "none", borderTop: "1px solid var(--border-subtle)" },
};

export function AiChatMarkdown({ content }: { content: string }) {
  const blocks = parseMarkdown(content);
  return (
    <div className="ai-chat-markdown" style={style.root}>
      {blocks.map((block, index) => {
        const key = `b${index}`;
        if (block.kind === "heading") {
          const Tag = (`h${Math.min(block.level + 2, 6)}`) as "h3" | "h4" | "h5" | "h6";
          return <Tag key={key} style={style.heading}>{renderInline(block.text, key)}</Tag>;
        }
        if (block.kind === "code") {
          return (
            <pre key={key} style={style.pre}>
              <code>{block.text}</code>
            </pre>
          );
        }
        if (block.kind === "rule") return <hr key={key} style={style.rule} />;
        if (block.kind === "quote") {
          return <blockquote key={key} style={style.quote}>{renderInline(block.text, key)}</blockquote>;
        }
        if (block.kind === "list") {
          const items = block.items.map((item, itemIndex) => (
            <li key={`${key}l${itemIndex}`}>{renderInline(item, `${key}l${itemIndex}`)}</li>
          ));
          return block.ordered
            ? (
              <ol key={key} start={block.start === 1 ? undefined : block.start} style={style.list}>
                {items}
              </ol>
            )
            : <ul key={key} style={style.list}>{items}</ul>;
        }
        if (block.kind === "table") {
          return (
            <div className="ai-chat-markdown__scroll" key={key} style={style.scroll}>
              <table style={style.table}>
                <thead>
                  <tr>
                    {block.header.map((cell, cellIndex) => (
                      <th key={`${key}h${cellIndex}`} style={style.head}>
                        {renderInline(cell, `${key}h${cellIndex}`)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {block.rows.map((row, rowIndex) => (
                    <tr key={`${key}r${rowIndex}`}>
                      {row.map((cell, cellIndex) => (
                        <td key={`${key}r${rowIndex}c${cellIndex}`} style={style.cell}>
                          {renderInline(cell, `${key}r${rowIndex}c${cellIndex}`)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        return <p key={key} style={style.paragraph}>{renderInline(block.text, key)}</p>;
      })}
    </div>
  );
}
