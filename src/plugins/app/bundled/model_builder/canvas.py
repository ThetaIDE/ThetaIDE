"""The drag-and-drop canvas: layer blocks wired into a stack.

Blocks are dragged in from the palette, connected port to port, and labelled
with the tensor shape they produce. A block whose shape does not line up is
outlined in red and carries the reason as its tooltip, so a broken stack can
be looked at rather than only rejected on save.

Every rule about what may connect to what lives in graph.py. This module
paints that graph and turns mouse gestures into calls on it.
"""

from __future__ import annotations

from PyQt6.QtCore import QMimeData, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QListWidget,
    QListWidgetItem,
)

from src.plugins.components.bundled.sequential_spec.sequential_spec import LAYER_KINDS
from frontend.theme import theme_color
from .graph import INPUT_KIND, GraphNode, ModelGraph

NODE_W = 208.0
NODE_H = 74.0
PORT_R = 6.5
MIME_LAYER = "application/x-theta-layer"

# Palette entries are grouped under these headings, in this order.
CATEGORY_ORDER = ("dense", "convolution", "activation", "pooling", "regularisation", "shape", "other")


def _summarise(node: GraphNode) -> str:
    """The one-line parameter summary shown under a block's title."""
    if node.is_input:
        return "dataset features"
    shown = [f"{k}={v}" for k, v in node.params.items() if v is not None and k != "bias"]
    return ", ".join(shown) if shown else "no parameters"


class NodeItem(QGraphicsItem):
    """One draggable block."""

    def __init__(self, node: GraphNode, canvas: GraphCanvas):
        super().__init__()
        self.node = node
        self.canvas = canvas
        self.error: str | None = None
        self.shape_label = ""
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not node.is_input)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setPos(node.x, node.y)
        self.setZValue(2)

    # ── Geometry ─────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        # Padded so the ports, which sit on the edges, are not clipped.
        return QRectF(-PORT_R, -PORT_R, NODE_W + 2 * PORT_R, NODE_H + 2 * PORT_R)

    def input_port(self) -> QPointF:
        return self.mapToScene(QPointF(NODE_W / 2, 0.0))

    def output_port(self) -> QPointF:
        return self.mapToScene(QPointF(NODE_W / 2, NODE_H))

    def port_at(self, scene_pos: QPointF) -> str | None:
        """Which port the given scene position is over, if any."""
        if not self.node.is_input and (scene_pos - self.input_port()).manhattanLength() <= PORT_R * 3:
            return "in"
        if (scene_pos - self.output_port()).manhattanLength() <= PORT_R * 3:
            return "out"
        return None

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.node.x, self.node.y = self.pos().x(), self.pos().y()
            self.canvas.refresh_edges()
        return super().itemChange(change, value)

    # ── Painting ─────────────────────────────────────────────────────────

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        body = QRectF(0, 0, NODE_W, NODE_H)

        fill = QColor(theme_color("raised" if self.node.is_input else "surface"))
        if self.error:
            border = QColor("#fb4934")
        elif self.isSelected():
            border = QColor(theme_color("accent"))
        else:
            border = QColor(theme_color("border"))

        path = QPainterPath()
        path.addRoundedRect(body, 6, 6)
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(border, 2.0 if (self.isSelected() or self.error) else 1.0))
        painter.drawPath(path)

        # A coloured spine marks the layer's category at a glance.
        if not self.node.is_input:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(self._category_color())))
            spine = QPainterPath()
            spine.addRoundedRect(QRectF(0, 0, 4, NODE_H), 2, 2)
            painter.drawPath(spine)

        title = QFont()
        title.setBold(True)
        title.setPointSize(11)
        painter.setFont(title)
        painter.setPen(QPen(QColor(theme_color("text"))))
        label = "input" if self.node.is_input else self.node.kind
        painter.drawText(QRectF(14, 8, NODE_W - 28, 20), Qt.AlignmentFlag.AlignLeft, label)

        small = QFont()
        small.setPointSize(9)
        painter.setFont(small)
        painter.setPen(QPen(QColor(theme_color("muted"))))
        # Elided rather than clipped: a block is narrow and a long parameter
        # list would otherwise be cut mid-word.
        summary = QFontMetrics(small).elidedText(_summarise(self.node), Qt.TextElideMode.ElideRight, int(NODE_W - 28))
        painter.drawText(QRectF(14, 29, NODE_W - 28, 18), Qt.AlignmentFlag.AlignLeft, summary)

        painter.setPen(QPen(QColor("#fb4934" if self.error else theme_color("secondary"))))
        painter.drawText(
            QRectF(14, 49, NODE_W - 28, 18),
            Qt.AlignmentFlag.AlignLeft,
            self.error and "shape error" or self.shape_label,
        )

        # Ports.
        painter.setPen(QPen(QColor(theme_color("border")), 1.0))
        painter.setBrush(QBrush(QColor(theme_color("accent"))))
        if not self.node.is_input:
            painter.drawEllipse(self.mapFromScene(self.input_port()), PORT_R, PORT_R)
        painter.drawEllipse(self.mapFromScene(self.output_port()), PORT_R, PORT_R)

    def _category_color(self) -> str:
        kind = LAYER_KINDS.get(self.node.kind)
        return {
            "dense": theme_color("primary"),
            "convolution": theme_color("secondary"),
            "activation": theme_color("accent"),
            "pooling": theme_color("number"),
            "regularisation": theme_color("focus"),
            "shape": theme_color("muted"),
        }.get(kind.category if kind else "other", theme_color("comment"))


class EdgeItem(QGraphicsPathItem):
    """A wire between two blocks."""

    def __init__(self, src: NodeItem, dst: NodeItem):
        super().__init__()
        self.src, self.dst = src, dst
        self.setZValue(1)
        self.setPen(QPen(QColor(theme_color("muted")), 2.0))
        self.update_path()

    def update_path(self) -> None:
        a, b = self.src.output_port(), self.dst.input_port()
        path = QPainterPath(a)
        # Vertical control points, so a wire leaves downward and arrives
        # upward however the blocks are arranged.
        lift = max(30.0, abs(b.y() - a.y()) * 0.45)
        path.cubicTo(QPointF(a.x(), a.y() + lift), QPointF(b.x(), b.y() - lift), b)
        self.setPath(path)


class LayerPalette(QListWidget):
    """The source side of the drag: one row per layer kind, grouped."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        self.setAlternatingRowColors(False)
        self._populate()

    def _populate(self) -> None:
        by_category: dict[str, list[str]] = {}
        for name, kind in LAYER_KINDS.items():
            by_category.setdefault(kind.category, []).append(name)

        ordered = list(CATEGORY_ORDER) + sorted(set(by_category) - set(CATEGORY_ORDER))
        for category in ordered:
            names = sorted(by_category.get(category, []))
            if not names:
                continue
            header = QListWidgetItem(category.upper())
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            font = header.font()
            font.setBold(True)
            font.setPointSize(8)
            header.setFont(font)
            header.setForeground(QBrush(QColor(theme_color("comment"))))
            self.addItem(header)
            for name in names:
                item = QListWidgetItem(f"   {name}")
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setToolTip(LAYER_KINDS[name].doc)
                self.addItem(item)

    def mimeData(self, items):  # type: ignore[override]
        data = QMimeData()
        for item in items:
            kind = item.data(Qt.ItemDataRole.UserRole)
            if kind:
                data.setData(MIME_LAYER, str(kind).encode())
                break
        return data


class GraphCanvas(QGraphicsView):
    """Scene, items, and the mouse gestures that edit the graph."""

    graph_changed = pyqtSignal()
    selection_changed = pyqtSignal(object)  # GraphNode | None

    def __init__(self, graph: ModelGraph, parent=None):
        super().__init__(parent)
        self.graph = graph
        self._items: dict[str, NodeItem] = {}
        self._edges: list[EdgeItem] = []
        self._pending: QGraphicsPathItem | None = None
        self._pending_src: NodeItem | None = None

        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setAcceptDrops(True)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setBackgroundBrush(QBrush(QColor(theme_color("base"))))
        self.scene().selectionChanged.connect(self._on_selection)
        self.rebuild()

    # ── Building the scene ───────────────────────────────────────────────

    def rebuild(self) -> None:
        """Recreate every item from the graph, keeping the selection."""
        selected = self.selected_node()
        self.scene().clear()
        self._items.clear()
        self._edges.clear()
        self._pending = None
        self._pending_src = None

        for node in self.graph.nodes.values():
            item = NodeItem(node, self)
            self._items[node.node_id] = item
            self.scene().addItem(item)

        for src_id, dst_id in self.graph.edges:
            src, dst = self._items.get(src_id), self._items.get(dst_id)
            if src and dst:
                edge = EdgeItem(src, dst)
                self._edges.append(edge)
                self.scene().addItem(edge)

        if selected and selected.node_id in self._items:
            self._items[selected.node_id].setSelected(True)
        self.refresh_labels()

    def refresh_edges(self) -> None:
        for edge in self._edges:
            edge.update_path()

    def refresh_labels(self) -> None:
        """Push current shapes and errors onto the blocks."""
        plans = self.graph.plan_by_node()
        input_item = self._items.get(self.graph.input_node.node_id) if self.graph.input_node else None
        if input_item:
            input_item.shape_label = "(B, " + ", ".join(str(d) for d in self.graph.input_shape) + ")"
            input_item.error = None
            input_item.setToolTip("The shape the loader hands over, before any layer.")

        for node_id, item in self._items.items():
            if input_item is not None and item is input_item:
                continue
            plan = plans.get(node_id)
            if plan is None:
                item.shape_label = "not connected"
                item.error = None
                item.setToolTip("Wire this block into the stack for it to be saved.")
            elif plan.ok and plan.out_shape is not None:
                item.shape_label = "(B, " + ", ".join(str(d) for d in plan.out_shape) + ")"
                item.error = None
                item.setToolTip(LAYER_KINDS[item.node.kind].doc)
            else:
                item.shape_label = ""
                item.error = plan.error
                item.setToolTip(plan.error or "")
        for item in self._items.values():
            item.update()
        self.graph_changed.emit()

    # ── Selection ────────────────────────────────────────────────────────

    def selected_node(self) -> GraphNode | None:
        for item in self.scene().selectedItems():
            if isinstance(item, NodeItem):
                return item.node
        return None

    def _on_selection(self) -> None:
        self.selection_changed.emit(self.selected_node())

    # ── Drops from the palette ───────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(MIME_LAYER):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(MIME_LAYER):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(MIME_LAYER):
            return
        kind = bytes(event.mimeData().data(MIME_LAYER)).decode()
        if kind not in LAYER_KINDS:
            return
        where = self.mapToScene(event.position().toPoint())
        node = self.graph.add_node(kind, where.x() - NODE_W / 2, where.y() - NODE_H / 2)

        # Dropping a block onto the free end of the stack is the common case,
        # so wire it up rather than making every drop a two-step gesture.
        tail = self._free_tail()
        if tail is not None:
            self.graph.connect(tail, node.node_id)

        self.rebuild()
        self._items[node.node_id].setSelected(True)
        event.acceptProposedAction()

    def _free_tail(self) -> str | None:
        """The last node in the chain, when its output is unconnected."""
        chain = self.graph.chain()
        tail = chain.nodes[-1] if chain.nodes else self.graph.input_node
        if tail is None:
            return None
        return tail.node_id if not any(a == tail.node_id for a, _ in self.graph.edges) else None

    # ── Wiring by dragging between ports ─────────────────────────────────

    def _node_at(self, scene_pos: QPointF) -> NodeItem | None:
        for item in self.scene().items(scene_pos):
            if isinstance(item, NodeItem):
                return item
        return None

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        for item in self._items.values():
            port = item.port_at(scene_pos)
            if port == "out":
                self._pending_src = item
                self._pending = QGraphicsPathItem()
                self._pending.setPen(QPen(QColor(theme_color("accent")), 2.0, Qt.PenStyle.DashLine))
                self._pending.setZValue(3)
                self.scene().addItem(self._pending)
                return
            if port == "in":
                # Dragging off a connected input detaches it, which is how a
                # block gets rewired without deleting it.
                for src_id, dst_id in list(self.graph.edges):
                    if dst_id == item.node.node_id:
                        self.graph.disconnect(src_id, dst_id)
                        self.rebuild()
                        return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pending is not None and self._pending_src is not None:
            a = self._pending_src.output_port()
            b = self.mapToScene(event.position().toPoint())
            path = QPainterPath(a)
            path.lineTo(b)
            self._pending.setPath(path)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pending is not None and self._pending_src is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            target = self._node_at(scene_pos)
            self.scene().removeItem(self._pending)
            src = self._pending_src
            self._pending = None
            self._pending_src = None
            if target is not None and target is not src:
                reason = self.graph.can_connect(src.node.node_id, target.node.node_id)
                if reason is None:
                    self.graph.connect(src.node.node_id, target.node.node_id)
                    self.rebuild()
                else:
                    self.refuse(reason)
            return
        super().mouseReleaseEvent(event)

    def refuse(self, reason: str) -> None:
        """Hook the dialog overrides to surface a rejected connection."""

    # ── Keyboard ─────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            removed = False
            for item in list(self.scene().selectedItems()):
                if isinstance(item, NodeItem) and not item.node.is_input:
                    self.graph.remove_node(item.node.node_id)
                    removed = True
            if removed:
                self.rebuild()
                return
        super().keyPressEvent(event)

    def tidy(self) -> None:
        self.graph.tidy()
        self.rebuild()
        self.fit()

    def fit(self) -> None:
        """Size the scene to the blocks, leaving their positions alone."""
        if self._items:
            self.setSceneRect(self.scene().itemsBoundingRect().adjusted(-60, -60, 60, 60))
