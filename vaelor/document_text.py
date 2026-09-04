"""Extract indexable plain text from an uploaded document, one function in.

RAG on AI Chat keyword-searches the text of an attached file, so every format
has to become plain text before it is chunked. The office formats a person
actually uses today - ``.docx``/``.xlsx``/``.pptx`` - are Office Open XML, which
is a ZIP of XML, so the standard library reads them here and they add no
dependency. PDF needs ``pypdf`` and legacy ``.xls`` needs ``xlrd`` (both pure
Python). The pre-2007 binary ``.doc``/``.ppt`` formats have no lightweight
extractor, so they are DECLINED with an actionable message rather than ingested
as binary garbage that would answer questions wrongly - the honest-degrade the
rest of the assistant already follows (VD-035).

Everything here parses UNTRUSTED bytes (``name``, ``media_type`` and the file
come straight from the upload body), so it is defensive on every axis an
adversarial review found:

* **A DTD is rejected at the parser, not by a byte test.** A DOCTYPE is what
  lets XML define entities that expand into gigabytes ("billion laughs"), and a
  substring test for ``<!DOCTYPE`` is defeated by encoding the member as UTF-16.
  Expat decodes first and then calls a handler that refuses any DOCTYPE, so the
  guard holds whatever the encoding. No legitimate Office file carries a DTD.
* **Decompression is bounded.** A ZIP member is read through a running byte
  budget (``ZipExtFile.read(n)`` decompresses only ``n`` bytes), so a small
  upload cannot decompress into gigabytes, and the total across the archive is
  capped near the upload size rather than at a multiple of it.
* **Extracted text is bounded.** PDF pages accumulate under the same char
  ceiling, and the final text is truncated to it, so a pathological file cannot
  exhaust memory before chunking.
* **Every read failure becomes a DocumentExtractionError**, so a corrupt
  member or a malformed PDF is a clean decline, never an uncaught 500. Empty
  extraction - a scanned, image-only PDF - is an error, not an empty document
  that would cite nothing.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ElementTree
import zipfile
import zlib
from typing import Callable, Dict
from xml.parsers import expat

#: The total decompressed bytes a single upload's ZIP may yield, and the total
#: characters any format may extract. Both sit just above the 10 MiB upload cap
#: so a legitimate document fits while a decompression/entity bomb is refused
#: before it is held in memory.
_MAX_ZIP_READ_BYTES = 20 * 1024 * 1024
_MAX_EXTRACTED_CHARS = 6_000_000

#: Magic bytes of formats that must never be read down the text path: a binary
#: file given a text extension (``report.docx`` renamed ``report.txt``) would
#: otherwise be decoded into garbage and indexed.
_BINARY_MAGIC = (b"PK\x03\x04", b"%PDF-", b"\xd0\xcf\x11\xe0")

#: Runs of blank lines and trailing spaces collapse so chunking sees clean text.
_BLANK_RUN = re.compile(r"[ \t]*\n[ \t]*")
_MANY_NEWLINES = re.compile(r"\n{3,}")


class DocumentExtractionError(Exception):
    """A document could not be turned into indexable text. The message is
    user-facing and says what to do instead (re-save, export, supply text)."""


class _UnsafeXmlDeclaration(Exception):
    """Raised from the expat DOCTYPE handler to abort a parse carrying a DTD."""


def _local_name(tag: str) -> str:
    """The tag without its XML namespace - ``{...}t`` -> ``t``."""
    return tag.rsplit("}", 1)[-1]


def _reject_doctype(*_args) -> None:
    raise _UnsafeXmlDeclaration()


def _parse_xml(data: bytes) -> ElementTree.Element:
    # Parse with a raw expat parser, not `ElementTree.XMLParser`, so the DOCTYPE
    # handler is reachable (ElementTree does not expose its expat handle on 3.14)
    # and fires whatever the byte encoding - expat decodes first, so a UTF-16
    # DTD is caught where a byte-substring test is not. A TreeBuilder turns the
    # events into the Element tree the extractors walk; `buffer_text` coalesces a
    # run's characters into one node. Without a DTD no custom entity can be
    # declared, so entity-expansion bombs cannot arise.
    builder = ElementTree.TreeBuilder()
    parser = expat.ParserCreate(namespace_separator="}")
    parser.buffer_text = True
    parser.StartDoctypeDeclHandler = _reject_doctype
    parser.StartElementHandler = lambda tag, attrs: builder.start(tag, attrs)
    parser.EndElementHandler = builder.end
    parser.CharacterDataHandler = builder.data
    try:
        parser.Parse(data, True)
        return builder.close()
    except _UnsafeXmlDeclaration:
        raise DocumentExtractionError(
            "This document contains an unsafe XML declaration and was not read.")
    except expat.ExpatError as error:
        raise DocumentExtractionError(
            "This document's internal XML could not be read.") from error


class _ZipReader:
    """A ZIP office file, read under a shared decompressed-byte budget.

    ``read`` decompresses at most the remaining budget plus one byte and refuses
    the archive if a member runs past it, so a compression bomb is caught while
    it is still being decompressed rather than after. Any decompression failure
    (a damaged member, a truncated stream) becomes a DocumentExtractionError.
    """

    def __init__(self, raw: bytes):
        try:
            self._archive = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as error:
            raise DocumentExtractionError(
                "This file is not a valid Office document.") from error
        self._remaining = _MAX_ZIP_READ_BYTES

    def namelist(self) -> list[str]:
        return self._archive.namelist()

    def read(self, name: str) -> bytes:
        try:
            with self._archive.open(name) as member:
                data = member.read(self._remaining + 1)
        except KeyError:
            return b""
        except (zipfile.BadZipFile, zlib.error, OSError, EOFError) as error:
            raise DocumentExtractionError(
                "This document could not be read; a part of it is damaged."
            ) from error
        if len(data) > self._remaining:
            raise DocumentExtractionError("This document is too large to read safely.")
        self._remaining -= len(data)
        return data


def _text_of(element: ElementTree.Element) -> str:
    """Join every text node whose local name is ``t`` - the run text of
    ``w:t`` (Word), ``a:t`` (PowerPoint) and ``t`` (shared strings)."""
    return "".join(
        node.text or ""
        for node in element.iter()
        if _local_name(node.tag) == "t"
    )


def _extract_plain(raw: bytes) -> str:
    """Decode a text upload, preferring UTF-8 and falling back so a Latin-1
    log or a BOM does not become an error."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_docx(raw: bytes) -> str:
    reader = _ZipReader(raw)
    lines = []
    members = ["word/document.xml"] + sorted(
        name for name in reader.namelist()
        if re.fullmatch(r"word/(?:header|footer)\d*\.xml", name))
    for member in members:
        data = reader.read(member)
        if not data:
            continue
        root = _parse_xml(data)
        for paragraph in root.iter():
            if _local_name(paragraph.tag) == "p":
                text = _text_of(paragraph).strip()
                if text:
                    lines.append(text)
    return "\n".join(lines)


def _extract_pptx(raw: bytes) -> str:
    reader = _ZipReader(raw)
    slides = sorted(
        (name for name in reader.namelist()
         if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
        key=lambda name: int(re.search(r"(\d+)", name).group(1)),
    )
    blocks = []
    for slide in slides:
        root = _parse_xml(reader.read(slide))
        lines = [
            _text_of(paragraph).strip()
            for paragraph in root.iter()
            if _local_name(paragraph.tag) == "p" and _text_of(paragraph).strip()
        ]
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _shared_strings(reader: _ZipReader) -> list[str]:
    data = reader.read("xl/sharedStrings.xml")
    if not data:
        return []
    root = _parse_xml(data)
    return [
        _text_of(item)
        for item in root
        if _local_name(item.tag) == "si"
    ]


def _extract_xlsx(raw: bytes) -> str:
    reader = _ZipReader(raw)
    strings = _shared_strings(reader)
    sheets = sorted(
        name for name in reader.namelist()
        if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))
    rows_out = []
    for sheet in sheets:
        root = _parse_xml(reader.read(sheet))
        for row in root.iter():
            if _local_name(row.tag) != "row":
                continue
            cells = []
            for cell in row:
                if _local_name(cell.tag) != "c":
                    continue
                kind = cell.get("t")
                if kind == "s":  # value is an index into the shared-string table
                    value = next(
                        (child.text for child in cell
                         if _local_name(child.tag) == "v"), None)
                    try:
                        index = int(value)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= index < len(strings):  # a negative index would wrap
                        cells.append(strings[index])
                elif kind == "inlineStr":
                    cells.append(_text_of(cell))
                else:  # number, date, boolean - keep the literal value as text
                    value = next(
                        (child.text for child in cell
                         if _local_name(child.tag) == "v"), None)
                    if value is not None:
                        cells.append(value)
            line = "\t".join(part for part in cells if part and part.strip())
            if line.strip():
                rows_out.append(line)
    return "\n".join(rows_out)


def _extract_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader  # noqa: PLC0415 - optional heavy import, kept local
        from pypdf.errors import PdfReadError
    except ImportError as error:  # pragma: no cover - dependency ships with Vaelor
        raise DocumentExtractionError(
            "PDF support is not installed on this appliance.") from error
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # a PDF encrypted with an empty owner password
            except Exception:  # noqa: BLE001 - any decrypt failure means we cannot read it
                raise DocumentExtractionError(
                    "This PDF is password-protected and could not be read.")
        pages = []
        total = 0
        for page in reader.pages:  # accumulate under the char ceiling, then stop
            body = (page.extract_text() or "").strip()
            if body:
                pages.append(body)
                total += len(body)
                if total >= _MAX_EXTRACTED_CHARS:
                    break
    except DocumentExtractionError:
        raise
    except (PdfReadError, Exception) as error:  # noqa: BLE001 - pypdf raises broadly on bad input
        raise DocumentExtractionError(
            "This PDF could not be read; it may be corrupt or unusual.") from error
    return "\n".join(pages)


def _extract_xls(raw: bytes) -> str:
    try:
        import xlrd  # noqa: PLC0415 - optional import, kept local
    except ImportError as error:  # pragma: no cover - dependency ships with Vaelor
        raise DocumentExtractionError(
            "Legacy .xls support is not installed on this appliance.") from error
    try:
        book = xlrd.open_workbook(file_contents=raw)
    except Exception as error:  # noqa: BLE001 - xlrd raises broadly on malformed input
        raise DocumentExtractionError(
            "This legacy .xls file could not be read.") from error
    rows_out = []
    total = 0
    for sheet in book.sheets():
        for index in range(sheet.nrows):
            cells = [str(value) for value in sheet.row_values(index)]
            line = "\t".join(cell for cell in cells if cell and cell.strip())
            if line.strip():
                rows_out.append(line)
                total += len(line)
                if total >= _MAX_EXTRACTED_CHARS:
                    return "\n".join(rows_out)
    return "\n".join(rows_out)


def _decline_legacy(_raw: bytes) -> str:
    raise DocumentExtractionError(
        "Legacy .doc and .ppt files (pre-2007) are not supported. Open the file "
        "and re-save it as .docx or .pptx, or export it to PDF, then attach that.")


#: Canonical formats, resolved from the filename first (media types lie) with
#: the media type as a fallback. Text formats decode; the office formats have
#: an extractor; the legacy binary formats decline with instructions.
_EXTENSION_KIND = {
    "pdf": "pdf",
    "docx": "docx", "xlsx": "xlsx", "pptx": "pptx", "xls": "xls",
    "doc": "legacy", "ppt": "legacy",
    "txt": "text", "md": "text", "markdown": "text", "json": "text",
    "csv": "text", "tsv": "text", "yaml": "text", "yml": "text", "log": "text",
    "text": "text", "rst": "text", "ini": "text", "conf": "text", "xml": "text",
}

_MEDIA_KIND = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.ms-excel": "xls",
    "application/msword": "legacy",
    "application/vnd.ms-powerpoint": "legacy",
}

_EXTRACTORS: Dict[str, Callable[[bytes], str]] = {
    "text": _extract_plain,
    "pdf": _extract_pdf,
    "docx": _extract_docx,
    "xlsx": _extract_xlsx,
    "pptx": _extract_pptx,
    "xls": _extract_xls,
    "legacy": _decline_legacy,
}

#: The formats a person can attach, for the UI and its accept filter. Kept here
#: beside the extractor table so the two cannot drift.
SUPPORTED_EXTENSIONS = tuple(sorted(
    ext for ext, kind in _EXTENSION_KIND.items() if kind != "legacy"))
BINARY_EXTENSIONS = ("pdf", "docx", "xlsx", "pptx", "xls")


def _classify(media_type: str, name: str) -> str:
    extension = name.rsplit(".", 1)[-1].lower() if "." in (name or "") else ""
    if extension in _EXTENSION_KIND:
        return _EXTENSION_KIND[extension]
    mapped = _MEDIA_KIND.get(str(media_type or "").split(";", 1)[0].strip().lower())
    if mapped:
        return mapped
    if str(media_type or "").lower().startswith("text/"):
        return "text"
    return ""


def _normalize(text: str) -> str:
    text = (text or "")[:_MAX_EXTRACTED_CHARS]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _BLANK_RUN.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def extract_text(raw: bytes, media_type: str = "", name: str = "") -> str:
    """Return the plain text of ``raw``, or raise :class:`DocumentExtractionError`.

    The format is resolved from ``name`` (extension) first and ``media_type``
    second. An unsupported or legacy format, a binary file wearing a text
    extension, and a document that yields no text all raise with a message
    meant for the person who attached the file.
    """
    kind = _classify(media_type, name)
    extractor = _EXTRACTORS.get(kind)
    if extractor is None:
        raise DocumentExtractionError(
            "This file type is not supported. Attach a PDF, a Word/Excel/"
            "PowerPoint file (.docx/.xlsx/.pptx), or a text, Markdown, CSV, "
            "JSON, YAML, or log file.")
    if kind == "text" and any((raw or b"").startswith(magic) for magic in _BINARY_MAGIC):
        raise DocumentExtractionError(
            "This looks like a binary document with a text extension. Attach it "
            "with its real extension (.pdf, .docx, .xlsx, .pptx).")
    text = _normalize(extractor(raw))
    if not text:
        raise DocumentExtractionError(
            "No readable text could be extracted. A scanned or image-only PDF "
            "has no embedded text to index - export a text-based version, or "
            "attach the text directly.")
    return text
