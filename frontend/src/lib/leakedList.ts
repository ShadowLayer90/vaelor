/**
 * A list literal that reached the reader, read back as a list.
 *
 * Agent results are free text produced by a model and assembled server-side,
 * and one of them arrived as `["...", '...', '...']` — brackets, commas and
 * mixed quotes included — printed verbatim into the card. Whatever produced
 * that, the reader must not be shown source syntax, so any surface that prints
 * one of these strings recognises the shape and renders the items instead.
 *
 * Deliberately strict: it only accepts a bracketed run of quoted strings. A
 * sentence that merely contains a bracket, a JSON object, or a list of numbers
 * is left exactly as written, because guessing at prose is worse than printing
 * it unchanged.
 */
export function leakedListItems(value: unknown): string[] | null {
  const text = String(value ?? "").trim();
  if (text.length < 4 || !text.startsWith("[") || !text.endsWith("]")) return null;
  const body = text.slice(1, -1);
  const items: string[] = [];
  let index = 0;
  const skipSpace = () => {
    while (index < body.length && /\s/.test(body[index])) index += 1;
  };
  skipSpace();
  if (index >= body.length) return null;
  while (index < body.length) {
    const quote = body[index];
    if (quote !== "'" && quote !== "\"") return null;
    index += 1;
    let item = "";
    while (index < body.length && body[index] !== quote) {
      if (body[index] === "\\" && index + 1 < body.length) {
        index += 1;
        item += body[index] === "n" ? "\n" : body[index] === "t" ? "\t" : body[index];
      } else {
        item += body[index];
      }
      index += 1;
    }
    // An unterminated quote means this is not a list literal after all.
    if (body[index] !== quote) return null;
    index += 1;
    if (item.trim()) items.push(item.trim());
    skipSpace();
    if (index >= body.length) break;
    if (body[index] !== ",") return null;
    index += 1;
    skipSpace();
    // A trailing comma is fine; anything else after it must be another item.
    if (index >= body.length) break;
  }
  return items.length ? items : null;
}
