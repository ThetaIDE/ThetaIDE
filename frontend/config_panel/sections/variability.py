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
    inherited_seed = int(resolved.get("seed", 1) or 1)
    defaults = {
        "training_runs": 1,
        "starting_seed": inherited_seed,
        "training_seeds": [inherited_seed],
        "evaluation_seed": None,
        "eval_episodes": int(resolved.get("eval_episodes", 0) or 0),
        "parallel_runs": 1,
    }

    def add_option(key):
        if (
            key == "training_seeds"
            and not explicit_seed_mode
            or key in ("training_runs", "starting_seed")
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
    inherited_seed = int(resolved.get("seed", 1) or 1)
    training_count = integer_field(
        int(value.get("training_runs") if value.get("training_runs") is not None else 1), minimum=1
    )
    start_seed = QLineEdit(
        str(value.get("starting_seed") if value.get("starting_seed") is not None else inherited_seed)
    )
    start_seed.setObjectName("variabilityStartingSeed")
    start_seed.setToolTip("First seed; subsequent runs use consecutive seeds")
    seeds = QLineEdit(", ".join(str(x) for x in (value.get("training_seeds") or [])))
    seeds.setObjectName("variabilityTrainingSeeds")
    seeds.setToolTip("Unique integer seeds, separated by commas")
    evaluation = QLineEdit(str(value.get("evaluation_seed", "")))
    evaluation.setPlaceholderText("Runner default")
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

    def set_start_seed():
        text = start_seed.text().strip()
        try:
            val = int(text)
        except ValueError:
            from ..editors import set_error

            set_error(start_seed, "Enter an integer seed")
            return
        self._on_nested_edited("variability", "starting_seed", val)

    def set_count(value_count):
        self._on_nested_edited("variability", "training_runs", value_count)

    def switch_seed_mode():
        import copy

        updated = copy.deepcopy(self.raw_data.get("variability") or {})
        if explicit_seed_mode:
            updated.update(training_seeds=None, training_runs=1, starting_seed=inherited_seed)
        else:
            try:
                first = int(start_seed.text())
                count = training_count.value()
                if not 1 <= count <= 100000:
                    from ..editors import set_error

                    set_error(
                        training_count,
                        "Keep large run counts in consecutive-seed mode; at most 100000 seeds can be expanded into a list.",
                    )
                    return
                generated = list(range(first, first + count))
            except ValueError:
                return
            updated.update(training_runs=None, starting_seed=None, training_seeds=generated)
        result = self.document.set(("variability",), updated)
        if result.changed:
            self._render_boxes()

    def set_evaluation():
        text = evaluation.text().strip()
        try:
            val = int(text) if text else None
        except ValueError:
            from ..editors import set_error

            set_error(evaluation, "Enter an integer seed or leave blank")
            return
        self._on_nested_edited("variability", "evaluation_seed", val)

    seeds.editingFinished.connect(set_seeds)
    start_seed.editingFinished.connect(set_start_seed)
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
        if value.get("starting_seed") is not None:
            form.add_parameter(
                ("variability", "starting_seed"),
                "Starting training seed",
                _parameter_row(start_seed, remove=self._parameter_remove(("variability", "starting_seed"))),
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
