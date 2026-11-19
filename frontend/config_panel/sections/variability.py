"""Configuration section composition using shared parameter controls."""

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QLineEdit,
)

from ...widgets import (
    integer_field,
)
from ..controls import (
    ConfigBox,
    ParameterForm,
    _make_param_adder_button,
    _make_subtle_tool_button,
    _parameter_header,
    _parameter_row,
)
from ..layout import _two_column_parameter_layout


def _render_variability_box(self, data, resolved):
    inherited = resolved.get("variability")
    explicit = data.get("variability")
    value = (
        {**(inherited if isinstance(inherited, dict) else {}), **explicit} if isinstance(explicit, dict) else inherited
    )
    if not isinstance(value, dict):
        return
    box = ConfigBox()
    defaults = {
        "training_runs": 1,
        "training_seeds": [],
        "evaluation_seed": None,
        "eval_episodes": int(resolved.get("eval_episodes", 0) or 0),
        "parallel_runs": 1,
    }

    def add_option(key):
        if (
            key == "training_seeds"
            and not explicit_seed_mode
            or key == "training_runs"
            and explicit_seed_mode
        ):
            switch_seed_mode()
        elif key == "eval_episodes":
            self._on_field_edited(key, defaults[key])
            QTimer.singleShot(0, self._render_boxes)
        else:
            self._on_generic_path_edited(("variability", key), defaults.get(key))
            QTimer.singleShot(0, self._render_boxes)

    available = [
        key
        for key in defaults
        if (key not in value or value[key] is None) and (key != "eval_episodes" or key not in data)
    ]
    header = _parameter_header(
        "Training and Evaluation Variability",
        add=_make_param_adder_button(
            "Add variability parameter", available, add_option, parent_widget=self, allow_custom=False
        ),
    )
    if "variability" in data:
        remove = _make_subtle_tool_button("trash", "Remove variability overrides", size=24, icon_size=14)
        remove.clicked.connect(lambda: self.remove_block("variability"))
        header.addWidget(remove)
    box.add_layout(header)
    form = ParameterForm()
    explicit_seed_mode = value.get("training_seeds") is not None
    training_count = integer_field(
        int(value.get("training_runs") if value.get("training_runs") is not None else 1), minimum=1
    )
    training_count.setToolTip("Each launch draws this many fresh training seeds and records them")
    seeds = QLineEdit(", ".join(str(x) for x in (value.get("training_seeds") or [])))
    seeds.setObjectName("variabilityTrainingSeeds")
    seeds.setPlaceholderText("Recorded seeds of the run to reproduce")
    seeds.setToolTip("Unique integer seeds, separated by commas. Set them only to reproduce a recorded run; "
                     "otherwise each launch draws fresh seeds.")
    evaluation = QLineEdit("" if value.get("evaluation_seed") is None else str(value["evaluation_seed"]))
    evaluation.setPlaceholderText("Drawn each launch")
    episodes = integer_field(int(data.get("eval_episodes", resolved.get("eval_episodes", 0)) or 0), minimum=0)
    episodes.setToolTip("0 disables evaluation where supported; adding this box does not change the inherited value")

    def set_seeds():
        text = seeds.text().strip()
        try:
            parsed = [int(x.strip()) for x in text.split(",") if x.strip()]
        except ValueError:
            from ..editors import set_error

            set_error(seeds, "Enter comma-separated integer seeds")
            return
        self._on_nested_edited("variability", "training_seeds", parsed)

    def set_count(value_count):
        self._on_nested_edited("variability", "training_runs", value_count)

    def switch_seed_mode():
        import copy

        updated = copy.deepcopy(self.raw_data.get("variability") or {})
        if explicit_seed_mode:
            updated.update(training_seeds=None, training_runs=max(1, len(value.get("training_seeds") or [])))
        else:
            updated.update(training_runs=None, training_seeds=[])
        result = self.document.set(("variability",), updated)
        if result.changed:
            self._render_boxes()

    def set_evaluation():
        text = evaluation.text().strip()
        try:
            val = int(text) if text else None
        except ValueError:
            from ..editors import set_error

            set_error(evaluation, "Enter an integer seed, or leave blank to draw one each launch")
            return
        self._on_nested_edited("variability", "evaluation_seed", val)

    seeds.editingFinished.connect(set_seeds)
    training_count.valueChanged.connect(set_count)
    evaluation.editingFinished.connect(set_evaluation)
    episodes.valueChanged.connect(lambda v: self._on_field_edited("eval_episodes", v))
    if explicit_seed_mode:
        form.add_parameter(
            ("variability", "training_seeds"),
            "Training seeds",
            _parameter_row(seeds, remove=self._parameter_remove(("variability", "training_seeds"))),
        )
    else:
        if value.get("training_runs") is not None:
            form.add_parameter(
                ("variability", "training_runs"),
                "Training runs per method",
                _parameter_row(training_count, remove=self._parameter_remove(("variability", "training_runs"))),
            )
    if "evaluation_seed" in value:
        form.add_parameter(
            ("variability", "evaluation_seed"),
            "Evaluation seed",
            _parameter_row(evaluation, remove=self._parameter_remove(("variability", "evaluation_seed"))),
        )
    if "eval_episodes" in data:
        form.add_parameter(
            ("eval_episodes",),
            "Episodes per evaluation",
            _parameter_row(episodes, remove=self._parameter_remove(("eval_episodes",))),
        )
    if "parallel_runs" in value:
        parallel = integer_field(int(value["parallel_runs"]), minimum=1)
        parallel.setObjectName("variabilityParallelRuns")
        parallel.setToolTip(
            "Maximum simultaneous training runs for this experiment. "
            "1 runs them sequentially. The workspace worker limit also applies."
        )
        parallel.valueChanged.connect(lambda v: self._on_nested_edited("variability", "parallel_runs", v))
        form.add_parameter(
            ("variability", "parallel_runs"),
            "Maximum parallel runs",
            _parameter_row(parallel, remove=self._parameter_remove(("variability", "parallel_runs"))),
        )
    extra = {key: val for key, val in value.items() if key not in defaults}
    self._render_generic_dict_contents(form, extra, ("variability",))
    box.setProperty("parameterColumns", 2)
    box.add_layout(_two_column_parameter_layout(form))
    self.boxes_layout.addWidget(box)
