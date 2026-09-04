import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(scriptDir, "..");
const stylesDir = path.join(frontendRoot, "src", "styles");
const entryStyle = path.join(frontendRoot, "src", "styles.css");
const baselinePath = path.join(scriptDir, "design-system-baseline.json");
const tokenFile = "src/styles/tokens-auth.css";
const runtimeNames = new Set([
  "--preset-color",
  "--lighting-brightness",
  "--lighting-color",
  "--lighting-opacity",
  "--lighting-speed",
  "--curve-height",
]);

const colorLiteralPattern = /#[0-9a-f]{3,8}\b|\b(?:rgb|rgba|hsl|hsla)\s*\([^)]*\)/gi;
const unsupportedColorFunctionPattern = /\b(?:lab|lch|oklab|oklch|color|device-cmyk|light-dark)\s*\(/gi;
const tokenPattern = /var\(\s*(--[A-Za-z0-9_-]+)/g;
const declarationPattern = /(?:^|(?<=[;{}\n]))\s*(--[A-Za-z0-9_-]+)\s*:/gm;
const cssDeclarationPattern = /(?:^|(?<=[;{}\n]))\s*([A-Za-z-]+)\s*:\s*([^;{}]+?)(?=;|})/gm;

const sourceRoot = path.join(frontendRoot, "src");
const forbiddenAliases = new Map([
  ["--font-data", "--font-family-data"],
  ["--font-mono", "--font-family-data"],
  ["--font-sans", "--font-family-sans"],
  ["--font-ui", "--font-family-sans"],
  ["--mono", "--font-family-data"],
  ["--font-size-caption", "--font-size-body-sm"],
  ["--text-xs", "--font-size-body-sm"],
  ["--bg-input", "--bg-surface"],
  ["--bg-page", "--bg-canvas"],
  ["--bg-panel", "--bg-surface-raised"],
  ["--surface", "--bg-surface"],
  ["--surface-raised", "--bg-surface-raised"],
  ["--surface-subtle", "--bg-hover"],
  ["--line", "--border-subtle"],
  ["--line-strong", "--border-strong"],
  ["--text", "--text-primary"],
  ["--text-tertiary", "--text-muted"],
  ["--text-faint", "--text-muted"],
  ["--cyan", "--info"],
  ["--green", "--success"],
  ["--orange", "--accent"],
  ["--red", "--danger"],
  ["--muted", "--text-muted"],
  ["--shadow-lg", "--shadow-modal"],
]);
// Responsive scale: compact, tablet, desktop, and wide desktop.
const supportedBreakpoints = new Set([720, 1024, 1200, 1440]);
const forbiddenAliasNames = new Set(forbiddenAliases.keys());

const spacingProperties = new Set([
  "margin", "margin-block", "margin-inline", "margin-top", "margin-right", "margin-bottom", "margin-left", "margin-start", "margin-end",
  "padding", "padding-block", "padding-inline", "padding-top", "padding-right", "padding-bottom", "padding-left", "padding-start", "padding-end",
  "gap", "row-gap", "column-gap",
  "inset", "inset-block", "inset-inline", "inset-top", "inset-right", "inset-bottom", "inset-left", "inset-start", "inset-end",
]);
const typographyProperties = new Set(["font-size", "font-family", "font", "line-height", "letter-spacing"]);

// These are the existing approved micro-scale values. Token-backed values are
// checked through the token reference instead of being counted as raw debt.
const supportedSpacingPixels = new Set([
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
  19, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 36,
]);
const supportedTypography = {
  "font-size": new Set([
    ".7rem", ".72rem", ".74rem", ".76rem", ".78rem", ".8rem", ".82rem", ".84rem", ".85rem", ".88rem", ".9rem",
    "1.02rem", "1.05rem", "1.1rem", "1.15rem", "1.2rem", "1.4rem",
    "14px", "15px", "18px", "19px", "21px", "22px", "24px", "25px", "26px", "27px", "28px", "40px",
    "clamp(1.25rem, 2vw, 1.6rem)", "clamp(1.35rem, 2vw, 1.8rem)", "clamp(1.8rem, 3vw, 2.5rem)",
    "clamp(22px, 2.2vw, 30px)", "clamp(22px, 2.4vw, 32px)", "clamp(24px, 3vw, 38px)",
    "clamp(26px, 3vw, 38px)", "clamp(26px, 3vw, 40px)", "clamp(26px, 4vw, 38px)",
    "clamp(28px, 3vw, 36px)", "clamp(2rem, 4vw, 3.4rem)", "clamp(30px, 2.4vw, 38px)",
    "clamp(32px, 4vw, 44px)", "clamp(40px, 5vw, 62px)", "clamp(42px, 8vw, 64px)",
  ]),
  "line-height": new Set(["1", "1.02", "1.1", "1.2", "1.3", "1.35", "1.4", "1.45", "1.5", "1.55", "1.6", "1.65", "1.72", "1.8"]),
  "letter-spacing": new Set(["0", "normal", "-.01em", "-.02em", "-.03em", "-.04em", "-.055em", "-.07em", ".03em", ".035em", ".04em", ".05em", ".06em", ".07em", ".08em", ".09em", ".1em", ".11em", ".12em", ".14em", ".18em"]),
  "font-family": new Set(["inherit"]),
};

function cssFiles() {
  return [entryStyle, ...fs.readdirSync(stylesDir)
    .filter((name) => name.endsWith(".css"))
    .sort()
    .map((name) => path.join(stylesDir, name))];
}

function relative(file) {
  return path.relative(frontendRoot, file).split(path.sep).join("/");
}

function withoutComments(text) {
  return text.replace(/\/\*[\s\S]*?\*\//g, (comment) => comment.replace(/[^\n]/g, " "));
}

function normalizeValue(value) {
  return value.replace(/\s*!important\s*$/i, "").trim().replace(/\s+/g, " ");
}

function canonicalTypographyValue(value) {
  return normalizeValue(value).replace(/(^|[\s/])(-?)0\.(\d+)(?=(?:rem|em)\b)/g, "$1$2.$3");
}

function declarations(file) {
  const source = withoutComments(fs.readFileSync(file, "utf8"));
  return [...source.matchAll(cssDeclarationPattern)].map((match) => {
    const property = match[1].toLowerCase();
    const value = normalizeValue(match[2]);
    const propertyOffset = match.index + match[0].indexOf(match[1]);
    return {
      file: relative(file),
      line: source.slice(0, propertyOffset).split("\n").length,
      property,
      value,
    };
  });
}

function colorDeclarations(files) {
  const records = [];
  for (const file of files) {
    for (const declaration of declarations(file)) {
      for (const match of declaration.value.matchAll(colorLiteralPattern)) {
        records.push({ ...declaration, literal: match[0].toLowerCase() });
      }
    }
  }
  return records;
}

function rawDeclarations(files, propertySet) {
  return files.flatMap((file) => declarations(file).filter(({ property }) => propertySet.has(property)));
}

function isRuntimeToken(name) {
  return runtimeNames.has(name);
}

function undefinedTokenErrors(declared, used) {
  const errors = [];
  for (const [name, locations] of used) {
    if (!declared.has(name) && !isRuntimeToken(name)) {
      errors.push(`undefined token ${name} (${[...new Set(locations)].join(", ")})`);
    }
  }
  return errors;
}

function sourceLine(source, offset) {
  return source.slice(0, offset).split("\n").length;
}

function breakpointErrors(files) {
  const errors = [];
  for (const file of files) {
    const source = fs.readFileSync(file, "utf8");
    for (const match of source.matchAll(/@media\s*\(max-width:\s*(\d+)px\)/g)) {
      const width = Number(match[1]);
      if (!supportedBreakpoints.has(width)) errors.push(`unsupported breakpoint ${relative(file)}:${sourceLine(source, match.index)} ${width}px; use 720, 1024, 1200, or 1440px`);
    }
  }
  return errors;
}

function cssStructureErrors(source, file) {
  const errors = [];
  const openings = [];
  let quote = "";
  let inComment = false;
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    const next = source[index + 1];
    if (inComment) {
      if (character === "*" && next === "/") {
        inComment = false;
        index += 1;
      }
      continue;
    }
    if (quote) {
      if (character === "\\") index += 1;
      else if (character === quote) quote = "";
      continue;
    }
    if (character === "/" && next === "*") {
      inComment = true;
      index += 1;
    } else if (character === "\"" || character === "'") {
      quote = character;
    } else if (character === "{") {
      openings.push(sourceLine(source, index));
    } else if (character === "}") {
      if (openings.length === 0) errors.push(`unexpected closing block in ${file}:${sourceLine(source, index)}`);
      else openings.pop();
    }
  }
  if (inComment) errors.push(`unclosed comment in ${file}`);
  if (quote) errors.push(`unclosed string in ${file}`);
  for (const line of openings) errors.push(`unclosed block in ${file}:${line}`);
  return errors;
}

function forbiddenAliasErrorsForSource(source, file) {
  const errors = [];
  const clean = withoutComments(source);
  const seen = new Set();
  const add = (name, offset, context) => {
    const key = `${file}:${sourceLine(clean, offset)}:${name}:${context}`;
    if (seen.has(key)) return;
    seen.add(key);
    errors.push(`forbidden compatibility token ${name} in ${file}:${sourceLine(clean, offset)} (${context}); use ${forbiddenAliases.get(name)}`);
  };
  for (const match of clean.matchAll(tokenPattern)) {
    if (forbiddenAliasNames.has(match[1])) add(match[1], match.index, "reference");
  }
  for (const match of clean.matchAll(declarationPattern)) {
    if (forbiddenAliasNames.has(match[1])) add(match[1], match.index, "declaration");
  }
  return errors;
}

function forbiddenAliasErrors(files) {
  return files.flatMap((file) => forbiddenAliasErrorsForSource(fs.readFileSync(file, "utf8"), relative(file)));
}

function sourceFiles(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(absolute);
    if (!entry.isFile() || !/\.(tsx|jsx)$/.test(entry.name) || entry.name.endsWith(".test.tsx") || entry.name.endsWith(".test.jsx")) return [];
    return [absolute];
  });
}

function jsxOpeningControls(source) {
  const controls = [];
  for (let index = 0; index < source.length; index += 1) {
    if (source[index] !== "<") continue;
    const match = source.slice(index).match(/^<(input|select|textarea|button)\b/);
    if (!match) continue;

    let cursor = index + match[0].length;
    let braceDepth = 0;
    let quote = "";
    for (; cursor < source.length; cursor += 1) {
      const character = source[cursor];
      if (quote) {
        if (character === quote && source[cursor - 1] !== "\\") quote = "";
        continue;
      }
      if (character === "\"" || character === "'") {
        quote = character;
        continue;
      }
      if (character === "{") {
        braceDepth += 1;
        continue;
      }
      if (character === "}" && braceDepth > 0) {
        braceDepth -= 1;
        continue;
      }
      if (character === ">" && braceDepth === 0) break;
    }
    if (cursor >= source.length) continue;
    controls.push({ tag: match[1], attributes: source.slice(index, cursor + 1), offset: index });
    index = cursor;
  }
  return controls;
}

function renderedControlErrorsForSource(source, file) {
  const errors = [];
  for (const control of jsxOpeningControls(source)) {
    const attributes = control.attributes;
    const type = attributes.match(/\btype\s*=\s*["']([^"']+)["']/i)?.[1]?.toLowerCase();
    if (type === "hidden") continue;

    const className = attributes.match(/\bclassName\s*=\s*(?:["']([^"']*)["']|\{([\s\S]*?)\})/);
    if (!className) {
      errors.push({ file, line: sourceLine(source, control.offset), tag: control.tag, reason: "missing className" });
    } else {
      const dynamicClass = className[2] ?? "";
      const hasSharedContract = dynamicClass.includes("joinClassNames(");
      const canRenderEmpty = (className[1] ?? "").trim() === ""
        || (!hasSharedContract && /(?:\?|:)\s*""/.test(dynamicClass));
      if (canRenderEmpty && !file.startsWith("src/components/ui/")) errors.push({ file, line: sourceLine(source, control.offset), tag: control.tag, reason: "className can render empty" });
    }

    if (/\bstyle\s*=/.test(attributes)) {
      errors.push({ file, line: sourceLine(source, control.offset), tag: control.tag, reason: "inline style bypasses token-backed control styling" });
    }
  }
  return errors;
}

function renderedControlErrors(files) {
  return files.flatMap((file) => renderedControlErrorsForSource(fs.readFileSync(file, "utf8"), relative(file)));
}

function blockedInlineStyleErrors(files) {
  return files.flatMap((file) => {
    const source = fs.readFileSync(file, "utf8");
    return [...source.matchAll(/\.setAttribute\(\s*["']style["']/g)].map((match) =>
      `CSP-blocked setAttribute(style) ${relative(file)}:${sourceLine(source, match.index)}; use React style or CSSOM setProperty`,
    );
  });
}

function primitiveMigrationErrors(files) {
  return files.flatMap((file) => {
    const fileName = relative(file);
    if (fileName.startsWith("src/components/ui/")) return [];
    const source = fs.readFileSync(file, "utf8");
    const violations = [
      ...source.matchAll(/<button\b/g),
      ...source.matchAll(/className=["'][^"']*(?<![A-Za-z0-9_-])button(?:--|\s|["'])/g),
      ...source.matchAll(/className=["'][^"']*(?<![A-Za-z0-9_-])notice(?:--|\s|["'])/g),
    ];
    return violations.map((match) =>
      `legacy rendered primitive ${fileName}:${sourceLine(source, match.index)}; use Button or Notice`,
    );
  });
}
function approvalKey(file, property, value) {
  return `${file}\u0000${property}\u0000${value.toLowerCase()}`;
}

function loadBaseline() {
  if (!fs.existsSync(baselinePath)) return null;
  return JSON.parse(fs.readFileSync(baselinePath, "utf8"));
}

function loadDecorativeApprovals(baseline, errors) {
  if (!baseline) return new Map();
  if (baseline.version !== 3) {
    errors.push(`unsupported design-system baseline version ${String(baseline.version)}; expected 3`);
  }
  if (Object.hasOwn(baseline, "legacyColorLiterals") || Object.hasOwn(baseline, "legacySpacingDeclarations") || Object.hasOwn(baseline, "legacyTypographyDeclarations")) {
    errors.push("count-based design-system baselines are not allowed; use exact decorativeColorApprovals only");
  }
  if (!Array.isArray(baseline.decorativeColorApprovals)) {
    errors.push("baseline must define decorativeColorApprovals as an array");
    return new Map();
  }

  const approvals = new Map();
  for (const [index, entry] of baseline.decorativeColorApprovals.entries()) {
    const hasFields = ["file", "property", "value", "reason"].every((field) => typeof entry?.[field] === "string" && entry[field].trim());
    if (!hasFields) {
      errors.push(`decorativeColorApprovals[${index}] must contain exact file/property/value/reason strings`);
      continue;
    }
    if (Object.hasOwn(entry, "count")) {
      errors.push(`decorativeColorApprovals[${index}] cannot use count; approvals are exact path/property/value/reason tuples`);
    }
    const file = entry.file.replaceAll("\\", "/");
    const property = entry.property.toLowerCase();
    const value = entry.value.toLowerCase();
    const key = approvalKey(file, property, value);
    if (approvals.has(key)) {
      errors.push(`duplicate decorative color approval ${file} ${property} ${value}`);
    } else {
      approvals.set(key, { file, property, value, reason: entry.reason.trim() });
    }
  }
  return approvals;
}

function checkColorApprovals(records, approvals, errors) {
  const matched = new Set();
  for (const record of records) {
    const key = approvalKey(record.file, record.property, record.literal);
    if (!approvals.has(key)) {
      errors.push(`unapproved semantic hardcoded color ${record.file}:${record.line} ${record.property}: ${record.literal}; add an exact decorative approval only when this is non-semantic`);
    } else {
      matched.add(key);
    }
  }
  return matched;
}

function isSupportedSpacingValue(value) {
  const normalized = normalizeValue(value);
  if (!normalized) return false;

  // var(), calc(), and clamp() are token-backed or fluid expressions. Raw
  // values outside those functions still have to be on the supported scale.
  const rawValue = normalized
    .replace(/var\([^)]*\)/gi, " ")
    .replace(/(?:calc|clamp)\([^)]*\)/gi, " ");
  const lengths = [...rawValue.matchAll(/-?(?:\d+(?:\.\d+)?|\.\d+)(px|rem|em|ch|ex|vh|vw|vmin|vmax|%)\b/gi)];
  const hasUnsupportedPixel = lengths.some(([, unit], index) => unit.toLowerCase() === "px" && !supportedSpacingPixels.has(Math.abs(Number(lengths[index][0].replace(/px$/i, "")))));
  return !hasUnsupportedPixel
    && !lengths.some(([, unit]) => ["rem", "em", "ch", "ex"].includes(unit.toLowerCase()));
}

function isSupportedFontShorthand(value) {
  const normalized = canonicalTypographyValue(value);
  if (normalized === "inherit" || normalized.includes("var(--")) {
    const rawSize = normalized.match(/(?:^|\s)(-?(?:\d+(?:\.\d+)?|\.\d+)(?:px|rem|em|%))(?=\/|\s|$)/i);
    if (rawSize && !supportedTypography["font-size"].has(rawSize[1])) return false;
    const rawLineHeight = normalized.match(/\/\s*(-?(?:\d+(?:\.\d+)?|\.\d+)(?:px|rem|em|%|$))/i);
    if (rawLineHeight?.[1] && !supportedTypography["line-height"].has(rawLineHeight[1])) return false;
    return true;
  }

  const rawSize = normalized.match(/(?:^|\s)(-?(?:\d+(?:\.\d+)?|\.\d+)(?:px|rem|em|%))(?=\/|\s|$)/i);
  if (!rawSize || !supportedTypography["font-size"].has(rawSize[1])) return false;
  const rawLineHeight = normalized.match(/\/\s*(-?(?:\d+(?:\.\d+)?|\.\d+)(?:px|rem|em|%|$))/i);
  return !rawLineHeight?.[1] || supportedTypography["line-height"].has(rawLineHeight[1]);
}

function isSupportedTypographyValue(property, value) {
  const normalized = canonicalTypographyValue(value);
  if (normalized.includes("var(--")) {
    return property === "font" ? isSupportedFontShorthand(normalized) : true;
  }
  if (property === "font") return isSupportedFontShorthand(normalized);
  return supportedTypography[property]?.has(normalized) ?? false;
}

function uniqueRecords(records, fields) {
  const inventory = new Map();
  for (const record of records) {
    const key = fields.map((field) => record[field]).join("\u0000");
    const current = inventory.get(key);
    if (current) current.count += 1;
    else inventory.set(key, { ...record, count: 1 });
  }
  return [...inventory.values()];
}

function runNegativeSelfTests() {
  const failures = [];
  const supportedSpacingCases = ["14px", "var(--space-4) 14px", "clamp(var(--space-4), 3vw, 28px)"];
  for (const value of supportedSpacingCases) {
    if (!isSupportedSpacingValue(value)) failures.push(`supported spacing value was rejected: ${value}`);
  }
  if (isSupportedSpacingValue("0.31rem")) failures.push("unsupported rem spacing was accepted");
  if (isSupportedSpacingValue("999px")) failures.push("unsupported pixel spacing was accepted");
  if (!isSupportedTypographyValue("font-size", "0.74rem")) failures.push("supported normalized font-size was rejected");
  if (!isSupportedTypographyValue("letter-spacing", "0.08em")) failures.push("supported normalized letter-spacing was rejected");
  if (!isSupportedTypographyValue("letter-spacing", "-0.02em")) failures.push("supported normalized negative letter-spacing was rejected");
  if (isSupportedTypographyValue("font-size", "17px")) failures.push("unsupported font-size was accepted");
  if (isSupportedTypographyValue("font", "700 17px/1.35 var(--font-data)")) failures.push("unsupported font shorthand was accepted");

  const undefinedErrors = undefinedTokenErrors(new Set(["--known"]), new Map([
    ["--known", ["demo.css"]],
    ["--missing", ["demo.css"]],
    ["--preset-color", ["demo.css"]],
  ]));
  if (undefinedErrors.length !== 1 || !undefinedErrors[0].includes("--missing")) failures.push("undefined non-runtime token was not rejected");

  const exactApproval = new Map([[approvalKey("src/styles/demo.css", "box-shadow", "rgb(0 0 0 / 20%)"), { reason: "test" }]]);
  const colorErrors = [];
  const colorMatches = checkColorApprovals([
    { file: "src/styles/demo.css", line: 1, property: "box-shadow", literal: "rgb(0 0 0 / 20%)" },
    { file: "src/styles/demo.css", line: 2, property: "box-shadow", literal: "rgb(0 0 0 / 21%)" },
  ], exactApproval, colorErrors);
  if (colorMatches.size !== 1 || colorErrors.length !== 1) failures.push("exact decorative color approval did not reject an unapproved tuple");

  const aliasErrors = forbiddenAliasErrorsForSource(":root {\n  --font-data: var(--font-family-data);\n  color: var(--font-data);\n}", "demo.css");
  if (aliasErrors.length !== 2 || !aliasErrors.every((error) => error.includes("--font-data"))) failures.push("forbidden compatibility alias was not rejected");

  const controlErrors = renderedControlErrorsForSource("<input />\n<button style={{ color: \"#fff\" }}>Bad</button>\n<textarea className=\"ui-control\" />", "demo.tsx");
  if (controlErrors.length !== 3 || !controlErrors.some((error) => error.reason === "missing className") || !controlErrors.some((error) => error.reason.includes("inline style"))) failures.push("unstyled or inline-styled rendered control was not rejected");
  if (renderedControlErrorsForSource("<input className=\"ui-control\" />", "demo.tsx").length !== 0) failures.push("token-backed rendered control was rejected");
  if (cssStructureErrors(".good { content: \"}\"; /* { */ }", "demo.css").length !== 0) failures.push("valid CSS structure was rejected");
  if (!cssStructureErrors("@media (test) { .bad { color: red; }", "demo.css").some((error) => error.includes("unclosed block"))) failures.push("unclosed CSS block was accepted");
  const schemaErrors = [];
  loadDecorativeApprovals({ version: 3, decorativeColorApprovals: [{ file: "demo.css", property: "background", value: "#fff", reason: "test", count: 1 }] }, schemaErrors);
  if (!schemaErrors.some((error) => error.includes("cannot use count"))) failures.push("count-based decorative approval was accepted");

  if (failures.length) throw new Error(`Design-system negative self-test failed: ${failures.join("; ")}`);
  console.log("Design-system negative self-tests passed: structure, token, alias, color-approval, control, supported-scale, and unsupported-value failures are deterministic.");
}

if (process.argv.includes("--negative-self-tests")) {
  runNegativeSelfTests();
  process.exit(0);
}

const files = cssFiles();
const legacyFiles = files.filter((file) => relative(file) !== tokenFile);
const componentFiles = sourceFiles(sourceRoot);
const declared = new Set();
const used = new Map();
const errors = [];

for (const file of files) {
  const rawSource = fs.readFileSync(file, "utf8");
  errors.push(...cssStructureErrors(rawSource, relative(file)));
  const source = withoutComments(rawSource);
  for (const match of source.matchAll(declarationPattern)) declared.add(match[1]);
  for (const match of source.matchAll(tokenPattern)) {
    const name = match[1];
    if (!used.has(name)) used.set(name, []);
    used.get(name).push(relative(file));
  }
  for (const match of source.matchAll(unsupportedColorFunctionPattern)) {
    errors.push(`unsupported color function ${match[0]} in ${relative(file)}`);
  }
}

errors.push(...undefinedTokenErrors(declared, used));
errors.push(...forbiddenAliasErrors(files));
errors.push(...breakpointErrors(files));
errors.push(...blockedInlineStyleErrors(componentFiles));
errors.push(...primitiveMigrationErrors(componentFiles));

const baseline = loadBaseline();
if (!baseline) {
  errors.push(`missing baseline ${relative(baselinePath)}; define exact decorativeColorApprovals after review`);
}
const approvals = loadDecorativeApprovals(baseline, errors);
const colors = colorDeclarations(legacyFiles);
const matchedApprovals = checkColorApprovals(colors, approvals, errors);
const unusedApprovals = [...approvals.keys()].filter((key) => !matchedApprovals.has(key));

const spacingDebt = rawDeclarations(legacyFiles, spacingProperties).filter(({ value }) => !isSupportedSpacingValue(value));
const typographyDebt = rawDeclarations(legacyFiles, typographyProperties).filter(({ property, value }) => !isSupportedTypographyValue(property, value));
const renderedControlDebt = renderedControlErrors(componentFiles);

for (const declaration of uniqueRecords(spacingDebt, ["file", "property", "value"])) {
  errors.push(`unsupported spacing value ${declaration.file}:${declaration.line} ${declaration.property}: ${declaration.value} (${declaration.count} occurrence${declaration.count === 1 ? "" : "s"})`);
}
for (const declaration of uniqueRecords(typographyDebt, ["file", "property", "value"])) {
  errors.push(`unsupported typography value ${declaration.file}:${declaration.line} ${declaration.property}: ${declaration.value} (${declaration.count} occurrence${declaration.count === 1 ? "" : "s"})`);
}

const countUnique = (records, fields) => new Set(records.map((record) => fields.map((field) => record[field]).join("\u0000"))).size;
for (const control of uniqueRecords(renderedControlDebt, ["file", "line", "tag", "reason"])) {
  errors.push(`rendered ${control.tag} control ${control.file}:${control.line} ${control.reason} (${control.count} occurrence${control.count === 1 ? "" : "s"}); use a token-backed shared control contract`);
}
console.log(`Design-system CSS check: ${files.length} files, ${declared.size} declared tokens, ${used.size} referenced tokens.`);
console.log(`Hardcoded color literals: ${colors.length} occurrences; ${countUnique(colors, ["file", "property", "literal"])} exact tuples; decorative approvals ${approvals.size} configured, ${matchedApprovals.size} matched, ${unusedApprovals.length} unused.`);
console.log(`Unsupported spacing debt: ${spacingDebt.length} occurrences across ${countUnique(spacingDebt, ["file", "property", "value"])} exact declarations.`);
console.log(`Unsupported typography debt: ${typographyDebt.length} occurrences across ${countUnique(typographyDebt, ["file", "property", "value"])} exact declarations.`);
console.log(`Rendered-control audit: ${componentFiles.length} production TSX/JSX files, ${renderedControlDebt.length} control violations.`);
if (unusedApprovals.length) {
  console.warn(`Residual decorative approvals (${unusedApprovals.length} unused):`);
  for (const key of unusedApprovals) console.warn(`- ${key.split("\u0000").join(" ")}`);
}

if (errors.length) {
  console.error(`Design-system check failed (${errors.length} issue${errors.length === 1 ? "" : "s"}):`);
  for (const error of errors) console.error(`- ${error}`);
  process.exitCode = 1;
} else {
  console.log("Design-system check passed: no undefined non-runtime tokens, forbidden aliases, unapproved semantic colors, unsupported spacing/type values, or rendered-control styling violations.");
}
