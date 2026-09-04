/**
 * Exporting an AI Chat conversation to a downloadable Markdown file.
 *
 * The Markdown build and the filename slug are pure so they can be tested
 * without a DOM; `downloadConversationMarkdown` is the thin browser glue that
 * turns them into a download. Kept out of `AiChat.tsx` so that component stays
 * under the module line ceiling and the export format has one tested home.
 */

/** A conversation turn as this export reads it — role and rendered content. */
export interface ExportMessage {
  role: string;
  content: string;
}

/** The conversation fields the export needs — its title and recorded model. */
export interface ExportConversation {
  title: string;
  model: string;
}

/** The exported Markdown: a title, the model byline, then each turn in order. */
export function conversationExportMarkdown(
  conversation: ExportConversation,
  messages: ExportMessage[],
): string {
  return [
    `# ${conversation.title}`, "",
    `Model: ${conversation.model || "Not recorded"}`, "",
    ...messages.flatMap((message) => [
      `## ${message.role === "user" ? "You" : "Vaelor AI"}`,
      "", message.content, "",
    ]),
  ].join("\n");
}

/** A filesystem-safe download name for a conversation title. */
export function conversationExportFilename(title: string): string {
  const slug = title.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase();
  return `${slug || "ai-chat"}.md`;
}

/** Build the Markdown for a conversation and hand the browser the download. */
export function downloadConversationMarkdown(
  conversation: ExportConversation,
  messages: ExportMessage[],
): void {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(
    new Blob([conversationExportMarkdown(conversation, messages)], { type: "text/markdown" }),
  );
  link.download = conversationExportFilename(conversation.title);
  link.click();
  URL.revokeObjectURL(link.href);
}
