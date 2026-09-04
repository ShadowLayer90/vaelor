/*
 * Prepare a file the reader attaches to AI Chat for the RAG document endpoint.
 *
 * Text formats are sent as UTF-8 text (`content`); binary formats - PDF and the
 * Office Open XML documents (.docx/.xlsx/.pptx) plus legacy .xls - are sent as
 * base64 bytes (`content_b64`) for the server to extract text from. Splitting
 * the two here keeps AiChat.tsx under the module-line ceiling and makes the
 * accepted-type rule testable in one place.
 */

const BINARY = /\.(pdf|docx|xlsx|pptx|xls)$/i;
const TEXT = /\.(txt|md|markdown|json|csv|tsv|ya?ml|log|rst|ini|conf|xml)$/i;

/** What the `<input type="file">` offers, and the message shown on a bad type. */
export const DOCUMENT_ACCEPT =
  ".pdf,.docx,.xlsx,.pptx,.xls,.txt,.md,.markdown,.json,.csv,.tsv,.yaml,.yml,.log,.rst,.ini,.conf,.xml";
export const SUPPORTED_DOCUMENT_HINT =
  "Attach a PDF, a Word/Excel/PowerPoint file (.docx/.xlsx/.pptx), or a text, "
  + "Markdown, CSV, JSON, YAML, or log file.";

export interface DocumentUploadBody {
  name: string;
  media_type: string;
  content?: string;
  content_b64?: string;
}

/** base64-encode bytes in chunks - `btoa(String.fromCharCode(...all))` blows the
 *  call stack on a multi-megabyte file, so feed it 32 KiB at a time. */
function toBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let index = 0; index < bytes.length; index += chunk) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunk));
  }
  return btoa(binary);
}

export function isBinaryDocument(fileName: string): boolean {
  return BINARY.test(fileName);
}

export function isSupportedDocument(fileName: string): boolean {
  return BINARY.test(fileName) || TEXT.test(fileName);
}

export async function prepareDocumentUpload(
  file: File,
  limitBytes: number,
): Promise<DocumentUploadBody> {
  if (file.size > limitBytes) {
    const mib = Math.round(limitBytes / (1024 * 1024));
    throw new Error(`This file exceeds the ${mib} MiB document limit.`);
  }
  if (!isSupportedDocument(file.name)) {
    throw new Error(SUPPORTED_DOCUMENT_HINT);
  }
  if (isBinaryDocument(file.name)) {
    return {
      name: file.name,
      media_type: file.type || "application/octet-stream",
      content_b64: toBase64(new Uint8Array(await file.arrayBuffer())),
    };
  }
  return {
    name: file.name,
    media_type: file.type || "text/plain",
    content: await file.text(),
  };
}
