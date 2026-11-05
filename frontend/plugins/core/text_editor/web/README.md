# Bundled editor assets

Theta loads `index.html` and `vendor/editor.bundle.js` locally through Qt WebEngine.
Node and npm are build-time tools; users do not need them to run the editor.

To rebuild after changing `editor.js` or upgrading dependencies, run in this directory:

```sh
npm ci
npm run build
```

The build also regenerates the third-party notices from the packages in the bundle.
Commit `package-lock.json`, the bundle, and `vendor/THIRD_PARTY_LICENSES.txt` together.
The package and lock files pin the editor build. The licenses file includes the notices
for all JavaScript packages used in the distributed bundle. No external scripts, fonts,
or CDN resources are loaded at runtime.

The native panel owns the file paths, file contents, save/close dialogs, and file watcher.
The renderer owns CodeMirror state, undo history, selection, and scroll position per tab.
Each text update includes the document ID and monotonic revision. Save and close operations
take a renderer snapshot before acting, so pending WebChannel edits cannot leave a save behind.

`EditorService.open_file(path, line=None, column=None)` is the host routing entry point.
Line and column are one-based. Providers register a callback with `EditorService.register`;
an unavailable default falls back to the system file association. Provider failures remain
with that provider rather than launching another application.
