/**
 * Vite's `?raw` suffix returns a module's text. Tests use it to assert
 * structural contracts — such as "every page heading is rendered from the
 * canonical destination registry" — without booting nine workspaces.
 */
declare module "*?raw" {
  const content: string;
  export default content;
}
