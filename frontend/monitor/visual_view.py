"""Native visual dispatch, completed snapshot playback and layered inspection."""
from __future__ import annotations

import json

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QComboBox, QGraphicsItem, QGraphicsScene,
                            QGraphicsView, QHBoxLayout, QPushButton, QVBoxLayout, QWidget)

from ..widgets import DoubleSpinBox as QDoubleSpinBox, ToggleSlider, label
from ..theme import theme_color
from .visual_data import VisualSnapshot, comparable, snapshots


def signed_color(value, scale):
    t = min(1., abs(float(value)) / max(1e-12, scale))
    return QColor.fromRgbF(.25 + .65*t if value < 0 else .15,
                          .25 if value < 0 else .3 + .55*t, .45 + .35*t)


class Canvas(QGraphicsView):
    def __init__(self):
        super().__init__(QGraphicsScene())
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMinimumHeight(300)
        self.setBackgroundBrush(QColor(theme_color("base")))

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        if .03 < self.transform().m11() * factor < 100:
            self.scale(factor, factor)
        event.accept()

    def fit(self):
        rect = self.scene().itemsBoundingRect().adjusted(-20, -20, 20, 20)
        self.scene().setSceneRect(rect)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)


def text(scene, x, y, content, width=None):
    item = scene.addText(str(content))
    item.setDefaultTextColor(QColor(theme_color("text")))
    item.setPos(x, y)
    if width is not None or len(str(content)) > 150:
        item.setTextWidth(width or 850)
    return item


def draw_matrix(canvas, snapshot, layer, state, **_):
    layers = snapshot.visual["data"]["layers"]
    chosen = next((row for row in layers if row["id"] == layer), layers[0])
    values = snapshot[chosen["array"]]
    if state >= 0:
        values = values[state:state+1]
    if values.ndim != 2 or not values.size:
        raise ValueError("No matrix measurements for this selection")
    finite = np.isfinite(values)
    scale = float(np.max(np.abs(values[finite]))) if finite.any() else 0.
    value_range = chosen.get("value_range")
    if value_range is not None:
        scale = max(abs(float(value)) for value in value_range)
    magnitude = np.clip(np.where(finite, np.abs(values), 0.) / max(scale, 1e-12), 0., 1.)
    negative = values < 0
    rgba = np.empty((*values.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 255*np.where(negative, .25+.65*magnitude, .15)
    rgba[..., 1] = 255*np.where(negative, .25, .3+.55*magnitude)
    rgba[..., 2] = 255*(.45+.35*magnitude)
    rgba[..., 3] = 255
    rgba[~finite] = (128, 128, 128, 255)
    image = QImage(rgba.data, values.shape[1], values.shape[0], values.shape[1]*4,
                   QImage.Format.Format_RGBA8888).copy()
    item = canvas.scene().addPixmap(QPixmap.fromImage(image).scaled(850, 450,
        Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation))
    row_axis, column_axis = chosen.get('row_axis', 'probe samples'), chosen.get('column_axis', 'saved IDs')
    item.setToolTip(f"{chosen['id']} · {chosen.get('coverage', '')}\nRows: {row_axis}; columns: {column_axis}")
    semantics = chosen.get("value_semantics", "Signed activation")
    unit = chosen.get("unit", "model units")
    legend = (f"range {value_range[0]:g} → {value_range[1]:g}; fixed scale" if value_range is not None
              else f"range ±{scale:.5g}; per-snapshot scale")
    text(canvas.scene(), 0, -80, f"{chosen['id']} · rows: {row_axis} · columns: {column_axis}\n"
         f"{semantics}, {unit}: blue + / red −; {legend}; gray = missing")
    columns = chosen.get("feature_ids", [])
    names = ", ".join(map(str, columns[:16])) + (" …" if len(columns) > 16 else "")
    rows = chosen.get('row_ids', [])
    row_names = ', '.join(map(str, rows[:24])) + (' …' if len(rows) > 24 else '')
    text(canvas.scene(), 0, 460, chosen.get("coverage", "") + (f"\nColumn order: {names}" if names else "")
         + (f"\nRow order: {row_names}" if row_names else ""))



def draw_distribution(canvas, snapshot, **_):
    values = snapshot[snapshot.visual["data"]["array"]].ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite distribution measurements")
    counts, bins = np.histogram(values, bins=40)
    for i, count in enumerate(counts):
        height = 300 * int(count) / max(1, int(counts.max()))
        canvas.scene().addRect(i*20, 300-height, 19, height, QPen(Qt.PenStyle.NoPen), QColor("#38bdf8"))
    text(canvas.scene(), 0, 310, f"Values {bins[0]:.5g} → {bins[-1]:.5g}; y: count; {len(values)} finite samples")


def draw_interventions(canvas, snapshot, threshold=0, **_):
    graph = snapshot.visual["data"]["graph"]
    nodes = graph["nodes"]
    positions = {node["id"]: (650, 220) if node["id"] == "behavior" else (0, i*48)
                 for i, node in enumerate(nodes)}
    scale = max([abs(edge["effect"]) for edge in graph["edges"]] or [1])
    for edge in graph["edges"]:
        if abs(edge["effect"]) < threshold:
            continue
        start, end = positions[edge["source"]], positions[edge["target"]]
        mediation = edge["kind"] == "measured_path_mediation"
        pen = QPen(QColor("#fbbf24" if mediation else "#38bdf8"), 1 + 5*abs(edge["effect"])/max(scale, 1e-12))
        pen.setStyle(Qt.PenStyle.DashLine if mediation else Qt.PenStyle.DotLine)
        item = canvas.scene().addLine(start[0]+150, start[1]+16, end[0]+150, end[1]+16, pen)
        item.setToolTip(json.dumps({**edge, "baseline": snapshot.metadata["details"].get("baseline"),
                                   "test": edge.get("test", snapshot.metadata["details"].get("mediation_scope"))}, indent=2))
    for node in nodes:
        x, y = positions[node["id"]]
        item = text(canvas.scene(), x, y, node.get("label", node["id"]))
        item.setToolTip(json.dumps(node, indent=2))
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
    text(canvas.scene(), 0, -100, "Intervention evidence · dotted blue: measured total effects; dashed gold: tested mediation/rescue\n"
         "Effects in action-score units, not physical connections. Hover for baseline and test.")
    text(canvas.scene(), 0, max(y for x, y in positions.values()) + 70, graph.get("interpretation", ""))


def draw_network(canvas, snapshot, mode, state, layer="", threshold=0, grouped=False, evidence=None, **_):
    graph = snapshot.visual["data"]["graph"]
    scene = canvas.scene()
    positions = {}
    layer_rows = {row["id"]: row for row in graph["layers"]}
    collapse = {row["id"]: grouped and row["id"] != layer for row in graph["layers"]}
    values_by_layer = {}
    for i, row in enumerate(graph["layers"]):
        values = snapshot[row["activation"]]
        values = values[state] if state >= 0 else values.mean(axis=0)
        values_by_layer[row["id"]] = values
        n = 1 if collapse[row["id"]] else row["units"]
        for unit in range(row["units"]):
            positions[(row["id"], unit)] = (i*220, 0 if collapse[row["id"]] else (unit-(n-1)/2)*34)
        text(scene, i*220-55, -max(100, row["units"]*17+75) if not collapse[row["id"]] else -100,
             f"{row['label']}\n{row['units']}/{row['total_units']} units\n{row['operation']}"
             + (f"\nactivation scale ±{max(1e-12, float(np.max(np.abs(values)))):.5g}" if mode == "Activations" else ""), width=190)
    numeric = []
    edges = []
    for connection in graph["connections"]:
        if connection["kind"] == "normalization":
            a = positions[(connection["source"], 0)]; b = positions[(connection["target"], 0)]
            item = scene.addLine(a[0], 0, b[0], 0, QPen(QColor("#a78bfa"), 2, Qt.PenStyle.DashLine))
            item.setToolTip(connection["dependency"])
            text(scene, (a[0]+b[0])/2-70, 30, "Coupled LayerNorm\n(full layer)")
            continue
        weights = snapshot[connection["weights"]] if connection["kind"] == "linear" else None
        sources = snapshot[connection["source_values"]]
        sources = sources[state] if state >= 0 else sources.mean(axis=0)
        # Collapsed layers carry no invented aggregate edge weight.
        if collapse[connection["source"]] or collapse[connection["target"]]:
            a = positions[(connection["source"], 0)]; b = positions[(connection["target"], 0)]
            item = scene.addLine(*a, *b, QPen(QColor("#64748b"), 2, Qt.PenStyle.DashLine))
            item.setToolTip("Collapsed connection group; expand layers to inspect individual weights")
            continue
        for src, dst in connection["edges"]:
            weight = float(weights[dst, src]) if weights is not None else None
            contribution = float(sources[src])*weight if weight is not None else None
            value = contribution if mode == "Local contributions" else weight
            if value is not None and abs(value) < threshold:
                continue
            numeric.append(abs(value or 0))
            edges.append((connection, src, dst, value, weight, contribution))
    scale = max(numeric or [1])
    for connection, src, dst, value, weight, contribution in edges:
        a, b = positions[(connection["source"], src)], positions[(connection["target"], dst)]
        pen = QPen(signed_color(value, scale) if value is not None else QColor("#64748b"),
                   .5+3*abs(value or 0)/max(scale, 1e-12))
        if value is None:
            pen.setStyle(Qt.PenStyle.DashLine)
        item = scene.addLine(*a, *b, pen)
        item.setOpacity(.25+.75*abs(value or 0)/max(scale, 1e-12) if value is not None else .7)
        item.setToolTip(f"{connection['source']}:{src} → {connection['target']}:{dst}\n"
                        f"weight={weight}; source × weight={contribution}\n"
                        "Contribution is to linear preactivation; bias and nonlinear operations are separate.")
    for row in graph["layers"]:
        values = values_by_layer[row["id"]]
        scale_nodes = max(1e-12, float(np.max(np.abs(values))))
        for unit in range(1 if collapse[row["id"]] else row["units"]):
            x, y = positions[(row["id"], unit)]
            fill = signed_color(values[unit], scale_nodes) if mode == "Activations" and not collapse[row["id"]] else QColor("#334155")
            item = scene.addEllipse(x-9, y-9, 18, 18, QPen(QColor("#cbd5e1")), fill)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            item.setData(0, f"{row['id']}:{unit}")
            bias = float(snapshot[row["bias"]][unit]) if "bias" in row else None
            item.setToolTip(f"{row['id']}:{unit}\nactivation={values[unit]:.6g}; bias={bias}\n{row['operation']}"
                            + (f"\nnormalization scale={float(snapshot[row['scale']][unit])}" if "scale" in row else ""))
            text(scene, x+12, y-12, f"{row['units']} units" if collapse[row["id"]] else str(unit))
    if evidence:
        destination = (len(graph["layers"])*220, 0)
        text(scene, destination[0], -60, "Measured behavioral effect\n(indirect; " + evidence["graph"].get("cohort", "held-out probes") + ")")
        for node in evidence["graph"]["nodes"]:
            module = node.get("module")
            unit = node.get("unit")
            if module is None or unit is None:
                continue
            absolute = node.get("absolute_module") or (f"{snapshot.metadata['target']}.{module}" if snapshot.metadata["target"] else module)
            point = positions.get((absolute, unit))
            if point is None or collapse.get(absolute):
                continue
            effect = node.get("effect", node.get("heldout_effect", 0.))
            if abs(effect) < threshold:
                continue
            item = scene.addEllipse(point[0]-13, point[1]-13, 26, 26, QPen(QColor("#fbbf24"), 2))
            item.setToolTip(f"Measured effect={effect}; baseline={evidence['baseline']}; selected={node.get('selected')}")
            arc = scene.addLine(*point, *destination, QPen(QColor("#fbbf24"), 1, Qt.PenStyle.DotLine))
            arc.setToolTip("Measured total effect on behavior; not a direct synaptic connection")
    coverage = graph["coverage"]
    magnitude = f"0 → {scale:.5g} (per snapshot)" if numeric else "hidden in collapsed groups or no saved edges"
    text(scene, 0, max(y for x,y in positions.values())+85,
         f"{mode} · blue + / red −; edge magnitude: {magnitude}\n"
         f"Activations: {'probe '+str(state) if state >= 0 else 'mean over fixed cohort'}; model units. "
         f"Layers {coverage['layers_recorded']}/{coverage['layers_total']}; edges {coverage['edges_recorded']}/{coverage['edges_total']}\n"
         f"{coverage['selection']}\n{graph['semantics']}")


RENDERERS = {"matrix": draw_matrix, "distribution": draw_distribution,
             "network_graph": draw_network, "intervention_graph": draw_interventions}


class VisualsView(QWidget):
    def __init__(self, parent=None, *, snapshot_loader=None):
        super().__init__(parent)
        self._snapshot_loader = snapshot_loader or VisualSnapshot
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        layout.addLayout(bar)
        self.run = QComboBox(); self.instance = QComboBox(); self.output = QComboBox()
        self.step = QComboBox(); self.compare = QComboBox()
        for title, widget in (("Run / method", self.run), ("Monitor", self.instance),
                              ("Output", self.output), ("Optimizer step", self.step), ("Compare", self.compare)):
            bar.addWidget(label(title, "muted")); bar.addWidget(widget)
        controls = QHBoxLayout(); layout.addLayout(controls)
        self.follow = ToggleSlider(); self.follow.setChecked(True)
        controls.addWidget(label("Follow latest")); controls.addWidget(self.follow)
        self.play = QPushButton("Play"); controls.addWidget(self.play)
        self.mode = QComboBox(); self.mode.addItems(["Weights", "Activations", "Local contributions", "Intervention evidence"])
        self.layer = QComboBox(); self.state = QComboBox()
        self.threshold = QDoubleSpinBox(); self.threshold.setDecimals(6); self.threshold.setRange(0, 1e9)
        self.grouped = ToggleSlider()
        for title, widget in (("Mode", self.mode), ("Layer", self.layer), ("Probe", self.state),
                              ("Min |edge|", self.threshold), ("Collapse other layers", self.grouped)):
            controls.addWidget(label(title, "muted")); controls.addWidget(widget)
        self.fit_button = QPushButton("Fit"); controls.addWidget(self.fit_button)
        self.caption = label("Rich outputs appear when a declared monitor publishes a completed snapshot.", "muted")
        self.caption.setWordWrap(True); layout.addWidget(self.caption)
        row = QHBoxLayout(); layout.addLayout(row, 1)
        self.canvas, self.comparison = Canvas(), Canvas()
        row.addWidget(self.canvas); row.addWidget(self.comparison); self.comparison.hide()
        self.references, self.current = [], []
        self._signature = None
        self._loaded_key = None
        self._selector = None
        self._lineage = None
        self._selections = {}
        self.timer = QTimer(self); self.timer.setInterval(900); self.timer.timeout.connect(self._advance)
        self.play.clicked.connect(self._play)
        self.fit_button.clicked.connect(lambda: (self.canvas.fit(), self.comparison.fit()))
        for combo in (self.run, self.instance, self.output):
            combo.currentIndexChanged.connect(self._choices)
        self.step.activated.connect(self._historical)
        self.step.currentIndexChanged.connect(self._draw)
        for combo in (self.compare, self.mode, self.layer, self.state):
            combo.currentIndexChanged.connect(self._draw)
        self.threshold.valueChanged.connect(self._draw); self.grouped.toggled.connect(self._draw)
        self.follow.toggled.connect(self._follow)

    @staticmethod
    def _options(combo, pairs, preferred=None):
        old = combo.currentData() if preferred is None else preferred
        combo.blockSignals(True); combo.clear()
        for title, value in pairs:
            combo.addItem(str(title), value)
        index = combo.findData(old)
        combo.setCurrentIndex(max(0, index)); combo.blockSignals(False)

    def show_lives(self, lives, *, refresh=False):
        refs = snapshots(lives)
        signature = tuple((r["run"], str(r["path"]), r["name"]) for r in refs)
        if signature == self._signature:
            if refresh:
                self._draw()
            return
        self._signature, self.references = signature, refs
        self._choices()

    def _choices(self, *_):
        self._options(self.run, [(key,key) for key in dict.fromkeys(r["run"] for r in self.references)])
        refs = [r for r in self.references if r["run"] == self.run.currentData()]
        self._options(self.instance, [(key,key) for key in dict.fromkeys(r["instance"] for r in refs)])
        refs = [r for r in refs if r["instance"] == self.instance.currentData()]
        names = {r["name"]: r["label"] for r in refs}
        self._options(self.output, [(title,key) for key,title in names.items()])
        self.current = [r for r in refs if r["name"] == self.output.currentData()]
        pairs = [(f"{r['updates']} · env {r['transitions']}", str(r["path"])) for r in self.current]
        preferred = str(self.current[-1]["path"]) if self.follow.isChecked() and self.current else None
        self._options(self.step, pairs, preferred)
        self._options(self.compare, [("None", ""), *pairs])
        self._draw()

    def _historical(self, *_):
        self.follow.setChecked(False)

    def _follow(self, enabled):
        if enabled:
            self._choices()

    def _play(self):
        if self.timer.isActive():
            self.timer.stop(); self.play.setText("Play")
        else:
            self.follow.setChecked(False); self.timer.start(); self.play.setText("Pause")

    def _advance(self):
        if self.step.count():
            self.step.setCurrentIndex((self.step.currentIndex()+1) % self.step.count())

    def hideEvent(self, event):
        self.timer.stop(); self.play.setText("Play")
        super().hideEvent(event)

    def _draw(self, *_):
        selected = {item.data(0) for item in self.canvas.scene().selectedItems() if item.data(0)}
        self.canvas.scene().clear(); self.comparison.scene().clear(); self.comparison.hide()
        reference = next((r for r in self.current if str(r["path"]) == self.step.currentData()), None)
        if reference is None:
            self.caption.setText("No completed visual snapshots for this selection.")
            return
        if self._selector is not None:
            self._selections[self._selector] = (self.layer.currentData(), self.state.currentData())
        selector = (reference["run"], reference["instance"], reference["name"])
        preferred_layer, preferred_state = self._selections.get(selector, ("", -1))
        self._selector = selector
        primary, other = None, None
        try:
            primary = self._snapshot_loader(reference)
            meta, visual = primary.metadata, primary.visual
            if self._lineage != meta.get("lineage"):
                selected.clear()
                preferred_layer, preferred_state = "", -1
            self._lineage = meta.get("lineage")
            data = visual["data"]
            layers = data.get("layers", data.get("graph", {}).get("layers", []))
            self._options(self.layer, ([("All layers", "")] if visual["kind"] == "network_graph" else [])
                          + [(row["id"], row["id"]) for row in layers], preferred_layer)
            sample_key = data.get("sample_ids", data.get("graph", {}).get("sample_ids"))
            ids = primary[sample_key] if sample_key else []
            self._options(self.state, [("Fixed cohort mean" if visual["kind"] == "network_graph" else "All probes", -1),
                                      *[(f"State {identity}", i) for i,identity in enumerate(ids)]], preferred_state)
            network = visual["kind"] == "network_graph"
            self.mode.setEnabled(network); self.grouped.setEnabled(network)
            self.state.setEnabled(bool(len(ids))); self.layer.setEnabled(bool(layers))
            renderer = RENDERERS.get(visual["kind"])
            if renderer is None:
                raise ValueError(f"No native renderer for {visual['kind']}")
            if network and self.mode.currentText() == "Intervention evidence":
                self._evidence(primary, reference)
            else:
                renderer(self.canvas, primary, **self._parameters())
            comparison = next((r for r in self.current if str(r["path"]) == self.compare.currentData()), None)
            if comparison and not (network and self.mode.currentText() == "Intervention evidence"):
                other = self._snapshot_loader(comparison)
                if not comparable(primary, other):
                    raise ValueError("Comparison unavailable: initialization lineage, topology, target or probe cohort differ")
                renderer(self.comparison, other, **self._parameters())
                self.comparison.show()
                text(self.comparison.scene(), 0, self.comparison.scene().itemsBoundingRect().top()-40,
                     f"Comparison optimizer step {other.metadata['optimizer_updates']}")
            for item in self.canvas.scene().items():
                if item.data(0) in selected:
                    item.setSelected(True)
            self.caption.setText(f"{reference['run']} · {meta['instance']} · target {meta['target'] or '(root)'} · "
                f"optimizer step {meta['optimizer_updates']} · environment step {meta['transitions']} · "
                f"{self.state.currentText()} · {visual['value_semantics']}\n"
                "Wheel: zoom · drag: pan · click: select · hover: values. Blue = positive; red = negative.")
            key = (reference["run"], reference["instance"], reference["name"], meta.get("lineage"), self.layer.currentData(), self.grouped.isChecked())
            if key != self._loaded_key:
                self.canvas.fit(); self._loaded_key = key
            if comparison:
                self.comparison.fit()
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            self.caption.setText(f"Visual unavailable: {exc}")
        finally:
            if primary is not None: primary.close()
            if other is not None: other.close()

    def _parameters(self):
        return dict(mode=self.mode.currentText(), layer=self.layer.currentData(), state=self.state.currentData(),
                    threshold=self.threshold.value(), grouped=self.grouped.isChecked())

    def _evidence(self, primary, reference):
        # Match saved measurements exactly; a nearby checkpoint is not the same model.
        matches = [r for r in self.references if r["run"] == reference["run"] and r["kind"] == "intervention_graph"]
        draw_network(self.canvas, primary, **{**self._parameters(), "mode": "Weights"})
        for ref in reversed(matches):
            evidence = self._snapshot_loader(ref)
            try:
                if any(evidence.metadata.get(key) != primary.metadata.get(key) for key in
                       ("model_state_sha256", "probe_set_sha256", "optimizer_updates")):
                    continue
                self.canvas.scene().clear()
                draw_network(self.canvas, primary, **{**self._parameters(), "mode": "Weights"},
                             evidence={"graph": {**evidence.visual["data"]["graph"], "nodes": [
                                 {**node, "absolute_module": f"{evidence.metadata['target']}.{node['module']}"
                                  if evidence.metadata["target"] else node.get("module")}
                                 if node.get("module") is not None else node
                                 for node in evidence.visual["data"]["graph"]["nodes"]]},
                                       "baseline": evidence.metadata["details"].get("baseline")})
                # Keep behavioral effects in a separate region with indirect, dotted links.
                offset = self.canvas.scene().itemsBoundingRect().bottom()+160
                # Re-render in a dedicated panel so physical edges cannot acquire effect widths.
                self.comparison.scene().clear()
                draw_interventions(self.comparison, evidence, threshold=self.threshold.value())
                self.comparison.show(); self.comparison.fit()
                text(self.canvas.scene(), 0, offset, f"Matched intervention monitor {evidence.metadata['instance']} · optimizer step {evidence.metadata['optimizer_updates']} · baseline {evidence.metadata['details'].get('baseline')}")
                return
            finally:
                evidence.close()
        text(self.canvas.scene(), 0, self.canvas.scene().itemsBoundingRect().bottom()+80,
             "No intervention evidence for this exact model state, optimizer step and probe bank.")
