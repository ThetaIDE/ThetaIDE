import {basicSetup} from "codemirror";
import {EditorState, Compartment, Prec} from "@codemirror/state";
import {EditorView, keymap} from "@codemirror/view";
import {indentWithTab, undo, redo} from "@codemirror/commands";
import {indentUnit, StreamLanguage, HighlightStyle, syntaxHighlighting} from "@codemirror/language";
import {openSearchPanel} from "@codemirror/search";
import {tags} from "@lezer/highlight";
import {python} from "@codemirror/lang-python";
import {yaml} from "@codemirror/lang-yaml";
import {json} from "@codemirror/lang-json";
import {markdown} from "@codemirror/lang-markdown";
import {toml} from "@codemirror/legacy-modes/mode/toml";

const documents = new Map();
const theme = new Compartment(), language = new Compartment(), wrapping = new Compartment(), readOnly = new Compartment();
let bridge, view, activeId = null, colors, wrap = false;

function languageFor(name) {
  if (/\.(py|pyw)$/i.test(name)) return python();
  if (/\.(yaml|yml)$/i.test(name)) return yaml();
  if (/\.(json|jsonl)$/i.test(name)) return json();
  if (/\.(md|markdown)$/i.test(name)) return markdown();
  if (/\.toml$/i.test(name)) return StreamLanguage.define(toml);
  return [];
}

function themeFor(c) {
  return [EditorView.theme({
    "&": {backgroundColor: c.base, color: c.text, fontSize: "13px"},
    ".cm-content": {fontFamily: "Menlo, Consolas, monospace", caretColor: c.accent, padding: "12px 0"},
    ".cm-line": {padding: "0 16px"},
    ".cm-cursor, .cm-dropCursor": {borderLeftColor: c.accent},
    ".cm-gutters": {backgroundColor: c.panel, color: c.muted, borderRight: `1px solid ${c.border}`},
    ".cm-activeLine, .cm-activeLineGutter": {backgroundColor: c.surface},
    "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": {backgroundColor: c.border},
    ".cm-panels, .cm-tooltip": {backgroundColor: c.panel, color: c.text, borderColor: c.border},
    ".cm-searchMatch": {backgroundColor: c.border, outline: `1px solid ${c.accent}`},
    ".cm-searchMatch.cm-searchMatch-selected": {backgroundColor: c.surface},
    ".cm-foldPlaceholder": {backgroundColor: c.surface, color: c.muted, borderColor: c.border},
    ".cm-textfield, .cm-button": {background: c.surface, color: c.text, border: `1px solid ${c.border}`},
    ".cm-tooltip-autocomplete > ul > li[aria-selected]": {backgroundColor: c.surface, color: c.accent}
  }, {dark: parseInt(c.base.slice(1, 3), 16) < 128}), syntaxHighlighting(HighlightStyle.define([
    {tag: tags.keyword, color: c.accent},
    {tag: [tags.string, tags.regexp], color: c.primary},
    {tag: [tags.number, tags.bool, tags.null], color: c.number},
    {tag: tags.comment, color: c.comment, fontStyle: "italic"},
    {tag: [tags.definition(tags.variableName), tags.propertyName], color: c.secondary},
    {tag: [tags.heading, tags.strong], color: c.accent, fontWeight: "bold"}
  ]))];
}

function command(action) {
  bridge.actionRequested(action);
  return true;
}

function createState(doc) {
  return EditorState.create({doc: doc.text, extensions: [
    basicSetup, indentUnit.of("    "),
    Prec.highest(keymap.of([
      {key: "Mod-s", run: () => command("save")},
      {key: "Mod-Shift-s", run: () => command("save_as")},
      {key: "Mod-o", run: () => command("open")},
      {key: "Mod-n", run: () => command("new")},
      {key: "Mod-w", run: () => command("close")},
      indentWithTab
    ])),
    theme.of(colors ? themeFor(colors) : []),
    language.of(languageFor(doc.name)),
    wrapping.of(wrap ? EditorView.lineWrapping : []),
    readOnly.of(EditorState.readOnly.of(!!doc.readOnly)),
    EditorView.updateListener.of(update => {
      if (!activeId) return;
      const current = documents.get(activeId);
      current.state = update.state;
      if (update.docChanged) {
        current.revision++;
        bridge.documentChanged(activeId, update.state.doc.toString(), current.revision);
      }
      if (update.selectionSet || update.docChanged) reportCursor();
    })
  ]});
}

function reportCursor() {
  const head = view.state.selection.main.head, line = view.state.doc.lineAt(head);
  // CodeMirror positions use UTF-16; the UI column counts Unicode code points.
  bridge.cursorChanged(line.number, Array.from(line.text.slice(0, head - line.from)).length + 1);
}

window.thetaEditor = {
  show(doc, line, column) {
    if (activeId) {
      const previous = documents.get(activeId);
      previous.state = view.state;
      previous.scroll = view.scrollSnapshot();
    }
    let current = documents.get(doc.id);
    if (!current) {
      current = {state: createState(doc), revision: doc.revision};
      documents.set(doc.id, current);
    }
    activeId = doc.id;
    view.setState(current.state);
    view.dispatch({effects: [theme.reconfigure(colors ? themeFor(colors) : []),
      wrapping.reconfigure(wrap ? EditorView.lineWrapping : []), language.reconfigure(languageFor(doc.name)),
      readOnly.reconfigure(EditorState.readOnly.of(!!doc.readOnly)),
      current.scroll || EditorView.scrollIntoView(0, {y: "start", x: "start"})]});
    if (line) {
      const row = view.state.doc.line(Math.max(1, Math.min(line, view.state.doc.lines)));
      const position = row.from + Array.from(row.text).slice(0, Math.max(0, (column || 1) - 1)).join("").length;
      view.dispatch({selection: {anchor: position}, scrollIntoView: true});
    }
    reportCursor();
    view.focus();
  },
  snapshot() {
    if (activeId) documents.get(activeId).state = view.state;
    return Array.from(documents, ([id, doc]) => ({id, text: doc.state.doc.toString(), revision: doc.revision}));
  },
  remove(id) {
    documents.delete(id);
    if (activeId === id) {activeId = null; view.setState(EditorState.create());}
  },
  replace(doc) {
    // Reload is an explicit action and starts a new history for that file.
    const wasActive = activeId === doc.id;
    if (wasActive) activeId = null;
    documents.set(doc.id, {state: createState(doc), revision: doc.revision});
    if (wasActive) this.show(doc);
  },
  configure(palette, lineWrap) {
    colors = palette; wrap = lineWrap;
    if (activeId) view.dispatch({effects: [theme.reconfigure(themeFor(colors)),
      wrapping.reconfigure(wrap ? EditorView.lineWrapping : [])]});
    document.body.style.background = colors.base;
  },
  action(name) {
    if (!activeId) return;
    if (name === "find") openSearchPanel(view);
    if (name === "undo") undo(view);
    if (name === "redo") redo(view);
    view.focus();
  }
};

new QWebChannel(qt.webChannelTransport, channel => {
  bridge = channel.objects.bridge;
  document.getElementById("editor").replaceChildren();
  view = new EditorView({state: EditorState.create(), parent: document.getElementById("editor")});
  bridge.ready();
});
