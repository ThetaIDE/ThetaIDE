"""Rich documentation data and articles covering Theta-IDE, a general-purpose ML, DL, and RL experimentation platform."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DocArticle:
    id: str
    title: str
    category: str
    summary: str
    html_content: str
    keywords: List[str]


ARTICLES: List[DocArticle] = [
    DocArticle(
        id="overview",
        title="Welcome & Architecture Overview",
        category="Getting Started",
        summary="High-level architecture of Theta-IDE, the execution pipeline, and general-purpose ML/DL/RL experimentation.",
        keywords=["overview", "architecture", "intro", "welcome", "reinforcement learning", "deep learning", "machine learning"],
        html_content="""
<h2>Welcome to Theta-IDE</h2>
<p><b>Theta-IDE</b> is a general-purpose development environment and experimentation platform for <b>Reinforcement Learning (RL)</b>, <b>Deep Learning (DL)</b>, <b>Machine Learning (ML)</b>, and <b>modular algorithm benchmarking</b>.</p>

<div style="background-color: rgba(254, 128, 25, 0.12); border-left: 4px solid #fe8019; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Core Philosophy:</b> Unify the entire research experimentation lifecycle into a single, cohesive workflow &mdash; from visual Hydra configuration composition and real-time training telemetry, to Slurm cluster job orchestration, interactive visualization analysis, and an extensible Component Hub.
</div>

<h3>Framework Architecture</h3>
<ul>
  <li><b>Hydra Configuration Layer (<code>in/config/</code>):</b> Declarative, three-tier hierarchical configuration defining environments, agents, models, cluster resources, and experiment recipes.</li>
  <li><b>In-process pipeline services (<code>src/app/services/</code>) &amp; job manager (<code>frontend/job_manager.py</code>):</b> The app composes and validates configs in-process. A detached runner owns training child processes and queue scheduling; the desktop reconnects to durable status, logs and metrics. There is no HTTP server to start.</li>
  <li><b>Training runtimes:</b> The built-in PyTorch Lightning driver (<code>src/app/train.py</code>) provides automated checkpointing, device acceleration (CUDA/MPS/CPU), and structured logging. Method plugins can declare their own runner entrypoint in each agent profile.</li>
  <li><b>Components from the Theta Hub:</b> A fresh clone provides the framework and generic IDE features without research algorithms, experiment recipes, trained results, or domain environments. Install the components required by an experiment from the <b>Theta Hub</b> (<a href="ide://components">Components</a> tab, globe button), or create your own with the <a href="authoring_plugins">Authoring Guide</a>. Each component supplies its own dependencies and integration points.</li>
  <li><b>Plugin System:</b> Core plugins (Documentation, TensorBoard, arXiv Papers, Visual Model Builder) ship with the IDE and have no effect when switched off; community plugins install from the Theta Hub.</li>
</ul>

<p>Diagrams of how these pieces talk to each other are in <a href="architecture">System Architecture</a>.</p>

<h3>Key Workflows</h3>
<ol>
  <li><b>Install components:</b> In the <a href="ide://components">Components</a> tab, open the Theta Hub and install the components required by your experiment.</li>
  <li><b>Configure:</b> Create or open an experiment in the <a href="ide://config">Experiment Config</a> pane: pick a paradigm and an environment, and add methods (an algorithm and a model each). See <a href="managing_experiments">Creating Experiments</a>.</li>
  <li><b>Execute:</b> Press <code>F5</code> to train on your own machine, or submit Slurm batch jobs to an HPC cluster (see <a href="cluster_slurm">Cluster Execution</a>).</li>
  <li><b>Monitor:</b> Watch reward curves, losses and every other logged metric live in the <a href="ide://monitor">Training Monitor</a>.</li>
  <li><b>Analyze:</b> Compare runs in the <a href="ide://results">Results</a> browser; convergence plots, loss plots and Markdown reports are generated at the end of each run.</li>
</ol>
""",
    ),
    DocArticle(
        id="architecture",
        title="System Architecture",
        category="Getting Started",
        summary="How the IDE, the training pipeline and the files between them fit together.",
        keywords=["architecture", "system", "diagram", "frontend", "backend", "pipeline", "process", "telemetry",
                  "qprocess", "codebase", "structure", "how it works"],
        html_content="""
<h2>System Architecture</h2>
<p>ThetaIDE has no server. The IDE imports the pipeline's services and calls them directly, and it runs training as child processes. Click a diagram to open it full size.</p>

<h3>Processes and the files between them</h3>
<p><a href="diagram:architecture_overview"><img src="diagram:architecture_overview" width="860"></a></p>
<p><i>Training never talks to the IDE except through its stdout. Everything else crosses through files. The highlighted path is the live one: the engine appends telemetry to <code>results/logs</code>, the IDE's file watcher picks it up, and the Monitor redraws. Runs started from the command line land in the same folders, so Results lists them too. The Theta Hub is the only network dependency.</i></p>

<h3>What happens when you launch training</h3>
<p><a href="diagram:launch_sequence"><img src="diagram:launch_sequence" width="860"></a></p>
<p><i>The config is composed twice through the same code (<code>src/app/pipeline/compose.py</code>): once in the IDE for the form and the pre-flight check (1&ndash;2), and again inside <code>run_pipeline.py</code> (5). That is why a recipe built in the IDE runs identically from the command line. Steps 7&ndash;9 repeat for as long as the engine trains.</i></p>

<h3>Codebase map</h3>
<table width="100%" cellspacing="0" cellpadding="8">
  <tr><th>Where</th><th>What it holds</th></tr>
  <tr><td><code>frontend/</code></td><td>The PyQt6 IDE: <code>app.py</code> (the window), the panels, <code>hub/</code> and <code>plugins/</code>. <code>local_backend.py</code>, <code>job_manager.py</code> and <code>live_run.py</code> are the bridge to the pipeline.</td></tr>
  <tr><td><code>run_pipeline.py</code></td><td>The command-line entry point, and what the IDE launches: <code>python run_pipeline.py &lt;group&gt;/&lt;experiment&gt; [overrides]</code>.</td></tr>
  <tr><td><code>src/app/pipeline/</code></td><td>Composition, pre-flight validation, engine dispatch (<code>engine.py</code>), the local runner, and which Python runs each engine (<code>runtime.py</code>).</td></tr>
  <tr><td><code>src/app/components/</code>, <code>src/app/workspace.py</code></td><td>Shared headless component installation/development/configuration services and paths for the current layout.</td></tr>
  <tr><td><code>src/app/services/</code></td><td>What the IDE calls in-process: <code>config_service</code>, <code>catalog</code>, <code>metrics</code>, <code>job_store</code>, <code>processes</code>, the exporters and the scaffolder.</td></tr>
  <tr><td><code>src/app/telemetry/</code></td><td>One logging API for every engine: <code>run.log()</code> writes <code>metrics.csv</code> and the <code>events.jsonl</code> live feed.</td></tr>
  <tr><td><code>src/app/core/</code></td><td>Model, plotter, data-module and evaluator registries, the paradigm loader, Lightning builder, callbacks and checkpoints. The canonical agent registry lives in <code>src/app/core/agent_registry.py</code>; old imports remain compatible.</td></tr>
  <tr><td><code>src/app/train.py</code>, component runner entry points</td><td>The built-in Lightning driver and runner entry points provided by installed components.</td></tr>
  <tr><td><code>src/usr/</code>, <code>in/config/</code></td><td>Authored components and active Hydra configs. Managed packages live in <code>src/plugins/components/installed</code>; bundled providers ship in <code>src/plugins/components/bundled</code>.</td></tr>
  <tr><td><code>plot/</code></td><td><code>manager.py</code> runs the selected research plotters, supplied by installed components such as a figures suite.</td></tr>
  <tr><td><code>results/</code>, <code>.thetaide/</code></td><td>Research outputs live under <code>results/</code>. IDE settings, run records, the queue (by default <code>.thetaide/runs/jobs/jobs.db</code>), installed GUI plugins, retained component snapshots and cache live under <code>.thetaide/</code>. These do not contain all active research source/configuration files.</td></tr>
</table>

<h3>Things that are easy to miss</h3>
<ul>
  <li><b>One worker thread.</b> Composition and the run catalog run on <code>LocalBackend</code>'s single background thread, because Hydra's global state is not thread-safe. Results come back to the window as callbacks.</li>
  <li><b>Files are the interface.</b> Apart from stdout, the IDE learns about a run only from <code>results/logs</code>.</li>
  <li><b>Two interpreters can be involved.</b> The IDE starts <code>run_pipeline.py</code> with its own Python, but <code>run_pipeline.py</code> starts each engine with <code>venv/bin/python3</code> when that exists (a site profile can name another venv with <code>venv_dir</code>). A project venv on an unsupported Python breaks training even when the IDE itself runs on a good one. On Windows a venv has no <code>bin/python3</code>, so the engine runs on the IDE's Python.</li>
  <li><b>Settings have two layers:</b> the workspace's <code>.thetaide/settings.toml</code> over <code>~/.config/thetaide/settings.toml</code>.</li>
</ul>
""",
    ),
    DocArticle(
        id="launching",
        title="Installing & Launching",
        category="Getting Started",
        summary="Set up the project environment, then start Theta-IDE on macOS, Linux, or Windows.",
        keywords=["install", "setup", "launch", "start", "run", "venv", "pipeline", "windows", "powershell", "gui"],
        html_content=r"""
<h2>Installing & Launching</h2>
<p>Theta-IDE runs from the same virtual environment as the training pipeline. Run every command from the <b>repository root</b>. See <code>docs/guides/INSTALL.md</code> for the supported setup and optional interface dependencies.</p>

<h3>1. One-Time Setup</h3>
<p><b>macOS / Linux:</b></p>
<pre><code>python3 -m venv venv
source venv/bin/activate
pip install -e ".[gui]"</code></pre>

<p><b>Windows (PowerShell):</b></p>
<pre><code>Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # once, so venvs can activate

py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e ".[gui]"</code></pre>

<h3>2. Start Theta-IDE</h3>
<pre><code>source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
python -m frontend</code></pre>
<p>That is the only command. The app composes configs in-process and runs training itself; there is no server to start first. The status bar shows <b>Pipeline ready</b> once the pipeline has loaded.</p>

<h3>3. Install Components</h3>
<p>The base IDE does not include research algorithms, experiment recipes, trained results, or domain environments. Open the <b>Theta Hub</b> from the <a href="ide://components">Components</a> tab and install the components required by your experiment. Components are installed into the workspace and can provide their own runners, environments, hooks, dependencies, and configuration.</p>

<h3>Launch Options</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Option</td>
    <td>Default</td>
    <td>Purpose</td>
  </tr>
  <tr><td><code>--data-dir &lt;path&gt;</code></td><td><code>.thetaide/runs/</code></td><td>Where run records and the job database are stored.</td></tr>
  <tr><td><code>--install-desktop-entry</code></td><td></td><td><b>Linux only:</b> add Theta-IDE to the application menu, then exit.</td></tr>
</table>

<h3>Reloading While Developing</h3>
<p>Press <code>Ctrl+R</code> in the terminal that launched the app to restart it with your latest code.</p>
<ul>
  <li><b>macOS / Linux:</b> works repeatedly.</li>
  <li><b>Windows:</b> works once per launch. Windows can't replace a running program in place, so the reload starts a fresh copy and the terminal prompt returns; the new window no longer listens to the terminal. It also needs a real console (PowerShell, Windows Terminal, or Command Prompt), not Git Bash.</li>
</ul>

<h3>If the Pipeline Is Unavailable</h3>
<p>The status bar shows <b>Pipeline unavailable</b> when the app isn't running from the project venv. You can still browse and edit configs (the preview shows an unvalidated local draft) and run <b>Start simulated demo</b> in the Queue tab (<code>Ctrl+F5</code>). Reinstall with <code>pip install -e ".[gui]"</code> and start the app again with <code>python -m frontend</code>.</p>

<h3>Stopping</h3>
<p>Closing or quitting the desktop disconnects it from the independent training runner. Active training and a started queue continue; reopen ThetaIDE to reconnect to durable status, logs and metrics. Use the run/queue cancellation controls to stop work explicitly. Queued jobs and runner state are stored under the app data directory (by default <code>.thetaide/runs/jobs/</code>, including <code>jobs.db</code>). TensorBoard and other desktop-owned plugins stop with the app.</p>
<p><b>Run notifications:</b> when a run completes, fails or is stopped, a toast appears in the window's bottom-right corner (&#10003; completed, &#10007; failed, &#9632; stopped, with how long it took) and a short chime plays, brighter for completed and lower for failed. <i>View results</i> on the toast plots the run in the Plot Viewer. If ThetaIDE isn't in front (minimised, in the tray, or another app has focus) the toast becomes a system notification instead. Switch the toast and the sound on or off in <i>Settings &rarr; Training pipeline &rarr; Run notifications</i>; <i>Try it</i> there shows a sample.</p>
""",
    ),
    DocArticle(
        id="managing_experiments",
        title="Creating, Renaming & Deleting Experiments",
        category="Getting Started",
        summary="Create new experiments and groups, rename or delete configs, and save or export your changes.",
        keywords=["new", "create", "rename", "delete", "duplicate", "experiment", "group", "save", "export", "config tree"],
        html_content=r"""
<h2>Creating, Renaming & Deleting Experiments</h2>
<p>Experiments live in <code>in/config/experiment/&lt;group&gt;/&lt;name&gt;.yaml</code>. Each group has a <code>_base.yaml</code> that sets the environment and paradigm its experiments inherit. Manage them from the config tree in the <a href="ide://config">Experiment Config</a> pane.</p>

<h3>Creating an Experiment</h3>
<ol>
  <li>Choose <i>File &rarr; New experiment</i> (<code>Ctrl+N</code>), click the new-experiment button in the tree's toolbar, or right-click a group and choose <b>New Experiment in this group&hellip;</b>.</li>
  <li>Select a group or subfolder, enter the experiment name in the tree, and press Enter.</li>
  <li>To start a <b>new group</b>, click the new-group button, enter its name in the tree, and press Enter. Then configure its group defaults.</li>
  <li>Add methods in the <b>Methods</b> box: each one names an algorithm (e.g. <code>sb3/ppo</code>) and a model (e.g. <code>basics/dnn</code>).</li>
</ol>
<p>Groups can hold subfolders (<b>New Subfolder&hellip;</b>), so a group can sort its experiments further. In the Components tab, right-click a component category and choose <b>New Component in this category&hellip;</b> to add an agent, model or environment config the same way.</p>

<h3>The Tree's Toolbar</h3>
<p>Above the tree: refresh, collapse all, expand all and filter; <b>Launch</b>, <b>Export</b> and the YAML preview toggle; and new group, new subfolder, new experiment and duplicate. Drag the buttons to rearrange them; right-click the toolbar and choose <b>Reset Toolbar Layout</b> to put them back.</p>

<h3>Right-Click Menu</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>On</td>
    <td>Actions</td>
  </tr>
  <tr><td>An experiment</td><td><b>Load in Config Viewer</b>, <b>Open in New Tab</b>, <b>Duplicate Experiment&hellip;</b>, <b>Rename&hellip;</b> (the <code>experiment_id</code> inside is updated to match), <b>Delete&hellip;</b></td></tr>
  <tr><td>A group</td><td><b>New Experiment in this group&hellip;</b>, <b>New Subfolder&hellip;</b>, <b>Export Experiment Plugin&hellip;</b> (the whole group as a Hub package), <b>Refresh Group</b>, <b>Delete Group&hellip;</b></td></tr>
  <tr><td>A subfolder</td><td><b>New Experiment&hellip;</b>, <b>New Subfolder&hellip;</b>, <b>Rename Folder&hellip;</b>, <b>Delete Folder&hellip;</b></td></tr>
  <tr><td>Anywhere</td><td><b>Collapse All</b> / <b>Expand All</b></td></tr>
</table>

<h3>Tabs</h3>
<p>Click an experiment to switch to it or preview it. Double-click it, or choose <b>Open in New Tab</b>, to open a new tab. <i>File &rarr; Close tab</i> (<code>Ctrl+W</code>) closes the current tab.</p>

<div style="background-color: rgba(251, 73, 52, 0.12); border-left: 4px solid #fb4934; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Protected base files:</b> A group's <code>_base.yaml</code> can't be deleted while experiments in that group still inherit from it, because Hydra could no longer load them. Delete or move those experiments first.
</div>
<p>Deleted files are removed from your working tree. If the file was committed to git, you can restore it with <code>git checkout -- &lt;path&gt;</code>.</p>

<h3>Saving & Exporting</h3>
<ul>
  <li><b>Save configuration</b> (<code>Ctrl+S</code>): write your edits back to the experiment's YAML file.</li>
  <li><b>Export draft YAML&hellip;</b> (<code>Ctrl+Shift+S</code>, or <b>Export</b> in the tree's toolbar) opens the export dialog, with three formats:
    <ul>
      <li><b>Standalone Recipe YAML (.yaml)</b>: one recipe file. Save it under <code>in/config/experiment/&lt;group&gt;/</code> and run it with <code>python run_pipeline.py &lt;group&gt;/&lt;name&gt;</code>.</li>
      <li><b>Theta Hub Component Archive (.zip)</b>: the recipe with everything it depends on, ready to publish on the Theta Hub (see <a href="authoring_plugins">Authoring Guide</a>).</li>
      <li><b>Standalone Git Repository / Directory</b>: the same files as a folder.</li>
    </ul>
  </li>
</ul>
""",
    ),
    DocArticle(
        id="ui_panels",
        title="IDE Panels & Navigation Guide",
        category="Interface",
        summary="Tour of the sidebar panels, from Components and Config to the Terminal, Console and plugin tabs.",
        keywords=["panels", "ui", "navigation", "sidebar", "tabs", "interface", "layout"],
        html_content="""
<h2>IDE Panels & Navigation</h2>
<p>Theta-IDE organizes its interface into vertical sidebar tabs. Each tab represents a specialized workstation for your experiments.</p>
<p>Some panels, such as Plots, TensorBoard and Queue, are hidden by default. Turn them on in <i>Settings &rarr; Appearance</i> (see <a href="sidebar_layout">Sidebar &amp; Layout</a>).</p>

<h3>1. Experiment Config (<code>config</code>)</h3>
<p>The primary workspace for configuring and launching experiments:</p>
<ul>
  <li><b>Config Tree:</b> Hierarchical browser mirroring <code>in/config/experiment/</code>: groups, their subfolders and experiments. Its toolbar launches, exports and creates experiments (see <a href="managing_experiments">Creating Experiments</a>).</li>
  <li><b>Experiment editor:</b> a form over the YAML, in tabs.
    <ul>
      <li><b>Experiment</b>: description, paradigm, environment and the run budget (timesteps or epochs, seed, evaluation, checkpoints). The <b>+</b> beside the heading adds any other setting.</li>
      <li><b>Methods</b>: one card per method, each with its algorithm, model and their parameters, plus optional <b>Shared Parameters</b> for every method. <b>Design...</b> beside the model opens the <b>Visual Model Builder</b>: drag layer blocks onto a canvas and wire them together, and it saves the network as a model config (a <code>sequential_spec</code> architecture) that you can pick like any other model.</li>
      <li><b>Add Overrides</b> adds a <b>Data Loaders</b>, <b>Evaluators</b> or <b>Figures and Tables</b> box to choose those components and their parameters, or <b>Checkpoint Settings</b> (the metric the best checkpoint is chosen by, <code>save_top_k</code> and how often to check).</li>
      <li>Opening a group's <code>_base.yaml</code> edits the <b>Group Defaults</b> its experiments inherit; inherited values are marked <i>(inherited)</i>. <i>Settings &rarr; Appearance &rarr; Experiment Panel Configuration</i> can show every resolved default too.</li>
    </ul>
  </li>
  <li><b>Hydra YAML Preview:</b> the composed configuration with syntax highlighting and validation errors. Toggle it from the tree's toolbar.</li>
</ul>

<h3>2. Training Monitor (<code>monitor</code>)</h3>
<p>Real-time telemetry and telemetry replay for live and finished training runs:</p>
<ul>
  <li>The monitor is for the run training now: it shows that run, or the most recent one when nothing is training. Past runs are viewed in the <b>Plot Viewer</b> (see below).</li>
  <li><b>Several runs training at once:</b> a <b>TRAINING NOW</b> row of chips appears above the status row, one per running experiment with how far along it is. Click a chip to show that run; Stop stops the run shown.</li>
  <li><b>Status row:</b> the run's status, name, paradigm and seed, with timing and training speed shown on each method's Queue card. A running job that goes quiet for two minutes shows <b>STALLED</b> and when it last reported; a finished run shows when it last wrote. Hover the name for the environment, methods and where the run's files are.</li>
  <li><b>System &middot; Logs &middot; Feed &middot; Files:</b> buttons at the end of the status row. Each opens that view in place of the charts; click it again to go back. The System button shows the training process's CPU use while it runs.</li>
  <li><b>Pipeline strip:</b> the run's methods in training order, one segment each (done, training with its progress, or waiting), with checkpoints &#9670;, the latest new best &#9733; and warnings &#9888; marked under it. Click a segment to show only that method's curves; click it again to show them all.</li>
  <li><b>Cards:</b> the metric the run is judged by, a second metric of your choice (pick it on the card), and the timestep budget.</li>
  <li><b>Charts:</b> a chart for every metric the run logs, in folding sections, one curve per agent: the evaluation section first, led by the headline chart (with its &plusmn;1 std band and the environment's maximum), losses four to a row. The evaluation section starts folded, since the headline card already shows its latest value: click its heading to show it, and it stays shown. While a run trains, its charts span its whole step budget (the part not reached yet is shaded), their y-axis only grows (a dashed line marks the highest value so far), and each curve still training has its newest point marked. <b>Hide</b> a chart or <b>Show all plots</b>. Hover over any chart: the inspector below lists every metric of every curve at that step. Values marked &asymp; come from the nearest other logged step (hover the cell to see which).</li>
  <li><b>Bottom bar:</b> the newest warning or error any method reported, with <b>Open log</b>; or, before there are metrics, what the run is waiting for (its place in the queue, the pipeline starting).</li>
  <li><b>System:</b> CPU, memory and GPU memory of the training process (and its child processes) over time, measured inside the training process itself.</li>
  <li><b>Logs:</b> the pipeline's output with search, an <i>Errors and warnings only</i> filter, and <i>Follow</i> to keep the newest lines in view.</li>
  <li><b>Feed:</b> what happened, newest first: run start and finish, checkpoints, new bests for <code>eval/</code> and <code>val/</code> metrics, NaN warnings and errors.</li>
  <li><b>Files:</b> checkpoints and other files the run reported; double-click to open one, or <b>Open run folder</b>.</li>
</ul>

<h3>3. Job Queue (<code>queue</code>)</h3>
<p>The Queue tab shows cards for processes that are currently training. Each card shows its experiment ID, description, status and elapsed time. Click a card to open that run in the Training Monitor.</p>
<ul>
  <li>Enable <b>Run methods in parallel</b> in Experiment Inputs to give each configured method its own concurrent training process and queue card. Clicking a method card opens the Training Monitor focused on that method.</li>
  <li>Launch the selected experiment with <code>F5</code>, stop the active run with <code>Shift+F5</code>, add an experiment to the queue with <code>Ctrl+Shift+Q</code>, and start or pause the queue with <code>Ctrl+Shift+R</code>. Start a simulated demo with <code>Ctrl+F5</code>.</li>
  <li>Queued jobs are saved under the app data directory, by default <code>.thetaide/runs/jobs/jobs.db</code>. The detached runner owns training and resource-based queue admission; closing the desktop does not cancel jobs. Reopen to reconnect, or cancel work explicitly.</li>
</ul>

<h3>4. Plots & Visualizations (<code>plots</code>)</h3>
<p>Inspect output plots auto-generated at the end of training pipelines:</p>
<ul>
  <li><b>Runs</b> (left): every run, for viewing past runs. Live and queued runs first, then recent ones; each row has a status mark, the run's name, how far along it is (<i>training &middot; 62%</i>, <i>done</i>, <i>queued</i>), a small reward curve and its latest reward. Click a run (or tick its box) to plot it; click again to take it off. The plot draws exactly the ticked runs, each in the color of its row, and <b>X</b> / <b>Y</b> offer every metric they log. <b>Clear plot</b> unticks them all.</li>
  <li><b>Open CSV / image&hellip;</b> shows a file from disk instead, such as a figure a run generated; ticking or unticking a run goes back to the runs.</li>
  <li><b>Zoom in</b>, <b>Zoom out</b> and <b>Fit</b> resize the view; <b>Export PNG&hellip;</b> saves what is shown.</li>
</ul>
<p>The figures and reports selected in the experiment configuration are generated at the end of a run under <code>results/plots/</code>; browse them in <a href="ide://results">Results &rarr; Figures</a>.</p>

<h3>5. Embedded Terminal (<code>terminal</code>)</h3>
<p>Built-in interactive terminal powered by xterm.js:</p>
<ul>
  <li>Full shell access with support for zsh, bash, and tmux.</li>
  <li>Terminal precedence mode: allows tmux prefix keys and terminal hotkeys to pass through uninterrupted.</li>
  <li><b>Every platform:</b> a pseudo-terminal on macOS and Linux, ConPTY on Windows (via <code>pywinpty</code>, which the <code>gui</code> extra installs there).</li>
  <li><b>Choosing the shell:</b> <i>Settings &rarr; Terminal</i>, or <code>shell</code> under <code>[terminal]</code> in <code>settings.toml</code> (e.g. <code>"pwsh -NoLogo"</code> or <code>"/bin/zsh -l"</code>). Empty picks one: PowerShell 7, then Windows PowerShell, then cmd on Windows; your login shell elsewhere. <code>cwd</code> sets the folder new shells start in. With <code>activate_venv = true</code> the project's <code>venv/</code> comes first on PATH.</li>
</ul>

<h3>6. Components &amp; the Theta Hub (<code>components</code>)</h3>
<p>Manage the building blocks experiments are made of:</p>
<ul>
  <li><b>Components Tree &amp; Viewer:</b> Browse component configurations, including agent profiles, environment definitions, model architectures, data loaders, plotters, paradigms and site profiles. Hub-installed components are read-only in the IDE; make a component copy to edit its code and configurations. Components are grouped by package.</li>
  <li><b>Terminal development:</b> <code>python run_pipeline.py component --help</code> exposes shared headless inspection, model copying, configuration editing/validation, and export. Deep Copy and Copy YAML use the same services as terminal commands. Copying a model creates an independent implementation; copying YAML keeps its implementation shared. Active files remain in the existing research/configuration directories.</li>
  <li><b>Theta Hub:</b> the globe button opens the Hub in this tab. Category tabs filter the component catalog; <b>Sort</b> orders the list and <b>Refresh</b> reloads the registry. Each card has <b>Install</b> or <b>Uninstall</b>. A suite can install a curated set of components together. <b>Back to Parameters</b> returns to the tree.</li>
  <li>When a component is uninstalled, its configuration files are hidden from the active tree and kept in the ignored <code>.thetaide/retained_configs/</code> store. Reinstalling that component restores its packaged profiles and user-created copies.</li>
</ul>

<h3>7. Results Browser (<code>results</code>)</h3>
<p>Every metric, value and file your runs produced: tick runs to chart every metric they logged, compare them, and open their tables, figures, reports and configs. See <a href="results_browser">Results Browser</a>.</p>

<h3>8. TensorBoard (<code>tensorboard</code>)</h3>
<p>An embedded TensorBoard dashboard for runs logged with <i>Log to TensorBoard</i>. See <a href="tensorboard_tab">TensorBoard Tab</a>.</p>

<h3>9. Console (<code>console</code>)</h3>
<p>Streams the pipeline's output and accepts a few quick commands. See <a href="console_pane">Console</a>.</p>

<h3>Plugin Tabs</h3>
<p>Core plugins add their own tabs: <b>arXiv Papers</b> (on by default; see <a href="arxiv_papers">arXiv Papers</a>), <b>TensorBoard</b> (above) and <b>Documentation</b> (this tab, opened with <code>F1</code>). The <b>Visual Model Builder</b> core plugin adds no tab: it adds the <b>Design...</b> button to each method's model in the Experiment tab. Switch any of them on or off in <i>Settings &rarr; Core Plugins</i>.</p>
""",
    ),
    DocArticle(
        id="learning_paradigms",
        title="Learning Paradigms & Constraints",
        category="Machine Learning",
        summary="The online RL, offline RL, supervised and unsupervised paradigms, and the rules each one enforces.",
        keywords=["paradigms", "online", "offline", "supervised", "unsupervised", "autoencoder", "ppo", "cql", "iql",
                  "sac", "dataset"],
        html_content="""
<h2>Learning Paradigms</h2>
<p>Theta-IDE strictly validates experiment compatibility using declared <b>paradigms</b>. Every experiment declares <code>paradigm: &lt;name&gt;</code>, which configures the training driver, callbacks, and validation rules.</p>

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Paradigm</td>
    <td>Allowed Agents / Models</td>
    <td>Evaluation Mechanism</td>
    <td>Key Constraints</td>
  </tr>
  <tr>
    <td><b><code>online_rl</code></b></td>
    <td>Online RL algorithms (e.g. <code>sb3/ppo</code>, <code>sb3/a2c</code>, <code>sb3/dqn</code>, <code>sb3/sac</code>)</td>
    <td>Fixed-episode live simulator rollouts via <code>EnvironmentEvaluatorCallback</code></td>
    <td>Needs an environment; budget in <code>total_timesteps</code>; can write transition datasets</td>
  </tr>
  <tr>
    <td><b><code>offline_rl</code></b></td>
    <td>Offline RL algorithms (e.g. <code>d3rlpy/cql</code>, <code>d3rlpy/iql</code>, <code>d3rlpy/bc</code>, <code>d3rlpy/td3_bc</code>)</td>
    <td>Replay buffer data module; validation loss and Bellman error via Lightning</td>
    <td><code>eval_episodes: 0</code>; requires a static transition dataset; budget in epochs</td>
  </tr>
  <tr>
    <td><b><code>supervised</code></b></td>
    <td>Predictive models only (default <code>mlp_classifier</code>)</td>
    <td>Classification metrics on held-out splits; early stopping and best checkpoint</td>
    <td>Algorithms forbidden; labeled static data; budget in epochs</td>
  </tr>
  <tr>
    <td><b><code>unsupervised</code></b></td>
    <td>Representation models only (default <code>dense_autoencoder</code>)</td>
    <td>Reconstruction loss (<code>val/loss</code>)</td>
    <td>Algorithms forbidden; unlabeled static data; budget in epochs</td>
  </tr>
</table>
<p>The rules live in <code>in/config/paradigms/&lt;name&gt;.yaml</code>: what each method needs, the training budget, the default data module and callbacks, and the constraints the pre-flight check enforces before a run starts.</p>

<h3>Transition Dataset Schema</h3>
<p>When online agents collect experience or offline datasets are loaded into replay buffers, transitions adhere to standard Gym / Gymnasium conventions:</p>

<h4>1. Core Standard RL Fields (Required &amp; Native)</h4>
<p>All standard Gym / Gymnasium environments and baseline offline datasets produce and consume the clean standard RL 5-tuple:</p>
<pre><code>{
  "obs": np.ndarray,            # Primary observation vector or image tensor
  "action": int | np.ndarray,   # Selected discrete action index or continuous action vector
  "reward": float,              # Scalar transition reward
  "next_obs": np.ndarray,       # Subsequent observation state
  "done": bool                  # Episode termination flag
}</code></pre>

<h4>2. Optional Domain-Specific Extensions</h4>
<p>Theta-IDE's open architecture allows community plugins and custom models (such as neurosymbolic hybrids, goal-conditioned agents, or multi-modal policies) to store or consume optional auxiliary fields via dataset adapters:</p>
<pre><code>{
  "logic_obs": np.ndarray,      # Optional: symbolic facts for logic reasoner plugins (e.g. BlendRL)
  "next_logic_obs": np.ndarray, # Optional: subsequent symbolic facts
  "info": dict                  # Optional: environment metadata or diagnostic flags
}</code></pre>

<div style="background-color: rgba(69, 133, 136, 0.12); border-left: 4px solid #458588; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Standard Environments vs. Plugin Extensions:</b>
  <ul style="margin: 6px 0 0 0; padding-left: 18px;">
    <li><b>Standard environments do NOT emit domain-specific logic natively:</b> Gym, Gymnasium, and standard benchmark environments produce clean observation vectors or images.</li>
    <li><b>Pure neural algorithms:</b> Baseline algorithms like PPO, IQL, and CQL train strictly on the core 5 fields and completely ignore auxiliary fields.</li>
    <li><b>Extensible adapters:</b> Specialized plugins (such as hybrid reasoners) compute relational groundings or custom representations on the fly via their own internal wrappers, leaving the core dataset schema clean and universal.</li>
  </ul>
</div>

<p>Online dataset generation via <code>DatasetWriter</code> automatically writes chunked <code>.pkl</code> archives (typically 100,000 transitions per chunk) accompanied by a <code>dataset_manifest.json</code> capturing git provenance, random seed, transition count, and environment metadata.</p>
""",
    ),
    DocArticle(
        id="blendrl_hybrid",
        title="BlendRL: Authoring a Composite Model Plugin",
        category="Machine Learning",
        summary="A current, code-oriented example of registering a model plugin that composes neural and symbolic child models.",
        keywords=["blendrl", "symbolic", "nsfr", "neumann", "neural", "prolog", "logic", "hybrid", "plugin", "component"],
        html_content="""
<h2>BlendRL: Authoring a Composite Model Plugin</h2>
<p>Use BlendRL as an example of a model component that owns its own composition. The model registry constructs one registered model class and passes its configuration to that class. It does not infer child-model slots from keys such as <code>neural</code> or <code>symbolic</code>. The parent plugin decides what those keys mean and builds each child itself.</p>

<div style="background-color: rgba(184, 187, 38, 0.12); border-left: 4px solid #b8bb26; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Status:</b> This is an authoring example, not a claim that a BlendRL implementation ships with a fresh checkout. Start with <b>File &rarr; Scaffold New Component Plugin</b> and choose <code>model</code>, or use the scaffold below as a template for your own registered components.
</div>

<h3>1. Let the parent model own composition</h3>
<p>A parent model can construct child models with <code>build_model()</code>. Each child must be registered in the model registry and return scores with a shape and meaning the parent can combine. Assign child <code>nn.Module</code>s as attributes so PyTorch includes their parameters in the parent model.</p>
<pre><code>import torch
import torch.nn as nn

from src.app.core.model_registry import build_model, register_model


@register_model("blendrl")
class BlendRLModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        n_actions: int = 2,
        neural: dict | str = "basics/dnn",
        symbolic: dict | str = "my_logic_head",
        blend_alpha: float = 0.25,
        env=None,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.neural = build_model(
            neural, env=env, device=device, obs_dim=obs_dim, n_actions=n_actions
        )
        self.symbolic = build_model(
            symbolic, env=env, device=device, obs_dim=obs_dim, n_actions=n_actions
        )
        self.blend_alpha = float(blend_alpha)
        if not 0.0 &lt;= self.blend_alpha &lt;= 1.0:
            raise ValueError("blend_alpha must be between 0 and 1")

    def forward(self, obs: torch.Tensor, **kwargs) -&gt; torch.Tensor:
        neural_scores = self.neural(obs, **kwargs)
        symbolic_scores = self.symbolic(obs, **kwargs)
        if neural_scores.shape != symbolic_scores.shape:
            raise ValueError("neural and symbolic model outputs must have the same shape")
        return torch.lerp(neural_scores, symbolic_scores, self.blend_alpha)</code></pre>
<p>This example treats the child outputs as compatible scores for illustration. A real logic branch must implement the domain's grounding and reasoning, and the selected training algorithm must accept the parent's output. If it needs extra inputs, define and wire those through the relevant agent, data module, or environment component; arbitrary dataset fields are not passed to a model automatically.</p>

<h3>2. Configure the component</h3>
<p>The scaffold creates the model package, a default file under <code>in/config/model/</code>, a <code>.theta_component.json</code> manifest, and a <code>pyproject.toml</code> entry point. Add nested child settings to the model's configuration and parameter schema:</p>
<pre><code># in/config/model/blendrl.yaml
architecture: blendrl
blend_alpha: 0.25
blend_function: softmax
blender:
  mode: neural
  model:
    name: basics/dnn
    hidden_sizes: [64, 64]
neural:
  name: basics/dnn
  hidden_sizes: [128, 128]
symbolic:
  name: my_logic_head
  rules: default</code></pre>
<p>Declare each child-model slot in the parent model's <code>parameter_schema</code> with <code>"x-theta-domain": "model"</code>. This makes the experiment editor show a model dropdown for the slot; after a model is selected, its declared parameters appear under the slot and can be added individually.</p>
<pre><code>"parameter_schema": {
  "type": "object",
  "properties": {
    "neural": {"type": "object", "x-theta-domain": "model"},
    "symbolic": {"type": "object", "x-theta-domain": "model"},
    "blender": {
      "type": "object",
      "required": ["mode", "model"],
      "properties": {
        "mode": {"type": "string", "enum": ["logic", "neural"]},
        "model": {"type": "object", "x-theta-domain": "model"}
      }
    },
    "blend_function": {"type": "string", "enum": ["softmax", "gumbel_softmax"]},
    "blend_alpha": {"type": "number", "default": 0.25}
  }
}</code></pre>

<h3>3. Use it from an experiment</h3>
<p>A method can select the parent model and provide its nested settings. These values reach the registered <code>BlendRLModel</code> constructor; the parent then resolves the child models:</p>
<pre><code>methods:
  cql_blendrl:
    agent: cql
    model:
      name: blendrl
      blend_alpha: 0.25
      neural:
        name: basics/dnn
        hidden_sizes: [128, 128]
      blender:
        mode: neural
        model:
          name: basics/dnn
          hidden_sizes: [64, 64]
      blend_function: softmax
      symbolic:
        name: my_logic_head
        rules: default</code></pre>

<h3>4. Check and package your plugin</h3>
<ol>
  <li>Confirm the package imports the module that contains <code>@register_model("blendrl")</code>; use <b>File &rarr; Reload Component Code</b> while developing.</li>
  <li>Install/register the child model components and confirm both child outputs satisfy the contract expected by the parent and chosen agent.</li>
  <li>Compose an experiment with <code>dry_run=true</code> before launching training.</li>
  <li>Include the model package, its config, manifest, dependencies, and any child components in the Theta Hub archive.</li>
</ol>
<p>The same pattern works for ensembles, multi-branch networks, and learned-plus-rule systems: the parent model defines the composition, validates its configuration, and returns the output expected by its consumer.</p>
""",
    ),
    DocArticle(
        id="hotkeys",
        title="Keyboard Shortcuts & Leader Chords",
        category="Workflow & Tools",
        summary="Complete reference for tmux-style leader key navigation and quick action hotkeys.",
        keywords=["hotkeys", "shortcuts", "leader", "tmux", "keyboard", "navigation", "launcher", "rofi", "palette"],
        html_content="""
<h2>Keyboard Shortcuts & Leader Navigation</h2>
<p>Theta-IDE features a high-efficiency <b>Leader key system</b> inspired by tmux and Vim. You can navigate between any panel instantly without taking your hands off the keyboard.</p>

<h3>The Launcher</h3>
<p>Press <code>Ctrl+Space</code> to open the launcher, a rofi-style search over every panel, plugin, settings page and menu command. Type part of a name (letters in order are enough: <code>tb</code> finds TensorBoard, <code>tr q</code> finds <i>Start or pause queue</i>), move with the arrow keys, <code>Tab</code>, <code>Ctrl+N</code>/<code>Ctrl+P</code> or <code>Ctrl+J</code>/<code>Ctrl+K</code>, and press <code>Enter</code> to open. <code>Esc</code>, a click outside or <code>Ctrl+Space</code> again closes it.</p>
<ul>
  <li>With nothing typed, what you opened most recently comes first.</li>
  <li>Opening a panel that is hidden in the sidebar shows it again. Plugins that are switched off are listed as <i>Turn on &hellip;</i>.</li>
  <li>Each row shows its shortcut, so the launcher also teaches the faster keys.</li>
  <li>Rebind it in <i>Settings &rarr; Hotkeys &rarr; Application Menu Shortcuts</i> (<code>open_launcher</code> under <code>[shortcuts]</code>).</li>
</ul>

<h3>The Action Key (Leader)</h3>
<p>Default: <code>Ctrl+B</code> (Customizable in <i>Settings &rarr; Hotkeys</i> to <code>Caps Lock</code>, <code>Alt</code>, <code>Ctrl</code>, or <code>Meta</code>).</p>
<p>Supports two operational modes:</p>
<ul>
  <li><b>Leader (Modal):</b> Tap the Action key, release it, and press a digit within 1.5 seconds.</li>
  <li><b>Chorded:</b> Hold the Action key and tap a digit simultaneously.</li>
</ul>

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Shortcut</td>
    <td>Target Pane / Action</td>
  </tr>
  <tr>
    <td><code>Action + 0</code></td>
    <td>Settings & Preferences</td>
  </tr>
  <tr>
    <td><code>Action + 1</code></td>
    <td>Components (configs and Theta Hub)</td>
  </tr>
  <tr>
    <td><code>Action + 2</code></td>
    <td>Experiment Configuration (Editor)</td>
  </tr>
  <tr>
    <td><code>Action + 3</code></td>
    <td>Training Monitor & Curves</td>
  </tr>
  <tr>
    <td><code>Action + 4</code></td>
    <td>Results Browser</td>
  </tr>
  <tr>
    <td><code>Action + 5</code></td>
    <td>Plot & Visualization Viewer</td>
  </tr>
  <tr>
    <td><code>Action + 6</code></td>
    <td>TensorBoard Dashboard</td>
  </tr>
  <tr>
    <td><code>Action + 7</code></td>
    <td>Job Queue Manager</td>
  </tr>
  <tr>
    <td><code>Action + 8</code></td>
    <td>Interactive Terminal</td>
  </tr>
  <tr>
    <td><code>Action + 9</code></td>
    <td>System Console Log</td>
  </tr>
</table>


<h3>Menu Shortcuts</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Shortcut</td>
    <td>Action</td>
  </tr>
  <tr><td><code>F1</code></td><td>Help &rarr; Documentation: open these docs</td></tr>
  <tr><td><code>Ctrl+Space</code></td><td>View &rarr; Open launcher</td></tr>
  <tr><td><code>Ctrl+N</code></td><td>File &rarr; New experiment</td></tr>
  <tr><td><code>Ctrl+Shift+P</code></td><td>File &rarr; Scaffold New Component Plugin</td></tr>
  <tr><td><code>Ctrl+Alt+R</code></td><td>File &rarr; Reload Component Code</td></tr>
  <tr><td><code>Ctrl+S</code></td><td>File &rarr; Save configuration</td></tr>
  <tr><td><code>Ctrl+Shift+S</code></td><td>File &rarr; Export draft YAML</td></tr>
  <tr><td><code>Ctrl+W</code></td><td>File &rarr; Close tab</td></tr>
  <tr><td><code>Ctrl+Q</code></td><td>File &rarr; Quit</td></tr>
  <tr><td><code>Ctrl++</code> / <code>Ctrl+-</code> / <code>Ctrl+0</code></td><td>View &rarr; Zoom In / Zoom Out / Reset Zoom</td></tr>
</table>

<h3>Training Shortcuts</h3>
<p>These work anywhere in the window; their buttons are in the <a href="ide://queue">Queue</a> tab.</p>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Shortcut</td>
    <td>Action</td>
  </tr>
  <tr><td><code>F5</code></td><td>Launch now: train the open experiment</td></tr>
  <tr><td><code>Shift+F5</code></td><td>Stop the active run</td></tr>
  <tr><td><code>Ctrl+F5</code></td><td>Start simulated demo (no training environment needed)</td></tr>
  <tr><td><code>Ctrl+Shift+Q</code></td><td>Add to queue</td></tr>
  <tr><td><code>Ctrl+Shift+R</code></td><td>Start or pause queue</td></tr>
</table>

<h3>Changing Shortcuts</h3>
<ul>
  <li>Rebind menu and training shortcuts in <i>Settings &rarr; Hotkeys &rarr; Application Menu Shortcuts</i>, or under <code>[shortcuts]</code> in <code>settings.toml</code> (for example <code>launch_training = "F6"</code>). The Hotkeys page warns when two actions share a shortcut, and <b>Restore default menu shortcuts</b> undoes your changes.</li>
  <li>Change the Action key, leader timeout and digit-to-pane mapping under <code>[hotkeys]</code>.</li>
</ul>

<h3>Reloading the App (Developers)</h3>
<p><code>Ctrl+R</code> is not an in-app shortcut. Pressed in the <b>terminal that launched Theta-IDE</b>, it restarts the app with your latest code changes. See <a href="launching">Installing &amp; Launching</a>.</p>
""",
    ),
    DocArticle(
        id="config_system",
        title="3-Tier Hierarchical Configuration",
        category="Getting Started",
        summary="How Hydra configuration files are structured across Tier 1 (Defaults), Tier 2 (Universal), and Tier 3 (Methods).",
        keywords=["config", "hydra", "tier", "yaml", "methods", "params", "hyperparameters"],
        html_content=r"""
<h2>3-Tier Hierarchical Configuration</h2>
<p>Configurations in Theta-IDE use a strict 3-tier hierarchy that eliminates parameter duplication while allowing granular per-method overrides.</p>

<h3>Tier 1: Defaults & Base Profiles</h3>
<p>Installed with each component, under <code>in/config/agent/&lt;package&gt;/&lt;algo&gt;.yaml</code> and <code>in/config/model/&lt;package&gt;/&lt;arch&gt;.yaml</code> (for example <code>agent/sb3/ppo.yaml</code> and <code>model/basics/dnn.yaml</code>). These define the baseline algorithm and model parameters (e.g. default batch size, discount factor \(\gamma\), network layer dimensions). Experiments refer to them by that path: <code>agent: sb3/ppo</code>, <code>model: basics/dnn</code>.</p>
<p>Each experiment group's <code>_base.yaml</code> sits between Tier 1 and the experiment: it sets the paradigm and environment, and anything its experiments share.</p>

<h3>Tier 2: Universal Experiment Parameters (<code>methods.params</code>)</h3>
<p>Universal scalars and hyperparameters applied across all methods in a single experiment. The experiment editor shows them as <b>Shared Parameters</b> in the Methods box:</p>
<pre><code>methods:
  params:
    epochs: 25
    gamma: 0.99
    lr: 3e-4</code></pre>

<h3>Tier 3: Method-Level Declarations</h3>
<p>Specific methods declared for execution inherit from Tier 1 and Tier 2, specifying only their differences:</p>
<pre><code>methods:
  ppo_dnn:
    agent: sb3/ppo
    model: basics/dnn
  ppo_tuned:
    agent: sb3/ppo
    model: basics/dueling_resnet
    lr: 1e-4
  blendrl_hybrid:
    agent: sb3/ppo
    model:
      name: blendrl
      blend_alpha: 0.25
      neural:
        name: basics/dnn
        hidden_sizes: [128, 128]
      symbolic:
        name: my_logic_head
        rules: default</code></pre>

<h3>Environment Keys</h3>
<p>Environment YAMLs (<code>in/config/env/&lt;package&gt;/&lt;name&gt;.yaml</code>, e.g. <code>env/gym_classic_control/cartpole.yaml</code>) define operational metadata declaratively. A group's <code>_base.yaml</code> picks one with <code>- override /env: gym_classic_control/cartpole</code>.</p>
<ul>
  <li><code>compatible_paradigms: [online_rl, offline_rl]</code> &mdash; the paradigms that may use it</li>
  <li><code>monitor_metric: "eval/reward" | "val/loss"</code> &mdash; target metric for checkpointing and tuning</li>
  <li><code>preprocess_on_load: true | false</code> &mdash; whether dataset requires offline conversion</li>
  <li>Figures and tables are selected in the experiment configuration.</li>
  <li>Simulator settings such as <code>env_id</code>, <code>num_envs</code> and <code>num_steps</code></li>
</ul>
""",
    ),
    DocArticle(
        id="cluster_slurm",
        title="Cluster Execution & Slurm Runner",
        category="Workflow & Tools",
        summary="How to submit jobs to HPC clusters using Slurm site profiles, resource limits, and email notifications.",
        keywords=["slurm", "cluster", "hpc", "ncshare", "arc", "sbatch", "gpu"],
        html_content="""
<h2>Cluster & Slurm Execution</h2>
<p>Theta-IDE supports seamless transitions between local testing and cluster-scale execution via Slurm.</p>

<h3>Site Profiles (<code>site</code>)</h3>
<ul>
  <li><code>site=local</code> (default): Interactive local execution inside subprocesses.</li>
  <li><code>site=ncshare</code> / <code>site=arc</code>: Automatically generates and dispatches Slurm batch scripts (<code>sbatch</code>).</li>
  <li><code>site=ncshare_gpu</code> / <code>site=ncshare_common</code>: NCShare's GPU partition (all methods in one job), or its CPU common partition (one job per method).</li>
</ul>
<p>Profiles live in <code>in/config/site/</code>. Add <code>consolidate=true</code> to put every method into one GPU job.</p>

<div style="background-color: rgba(251, 73, 52, 0.12); border-left: 4px solid #fb4934; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Cluster Push Mandate:</b> Always commit and push all code changes to GitHub before submitting cluster jobs, as remote compute nodes sync directly from the repository.
</div>

<h3>Configuring Cluster Resources</h3>
<p>Specify compute requirements directly in your experiment YAML:</p>
<pre><code>resources:
  time: "04:00:00"   # Wall clock limit (hh:mm:ss)
  gpus: 1            # Number of GPU accelerators
  cores: 16          # CPU cores allocated
  memory: "32G"      # RAM allocation</code></pre>
<p>Priority order: CLI flags &gt; experiment YAML &gt; site default config &gt; fallback defaults.</p>

<h3>Automated Slurm Pipeline Features</h3>
<ul>
  <li><b>Dependency Chaining:</b> Online data generation jobs automatically establish Slurm dependency holds (<code>--dependency=afterok:&lt;job_id&gt;</code>) on downstream offline comparison jobs.</li>
  <li><b>Status Notifications:</b> Configured with <code>--mail-type=END,FAIL</code> for immediate progress updates (a site profile can change it with <code>mail_type</code>).</li>
</ul>
<p>Runs that finish on the cluster land in the same <code>results/</code> folders, so once they are synced back the <a href="ide://results">Results</a> browser lists them alongside local runs.</p>
""",
    ),
    DocArticle(
        id="plugins_guide",
        title="Core Plugins & Community Extensions",
        category="Extensibility",
        summary="Guide to creating and managing plugins: Core vs Community plugins, lifecycle hooks, and the Plugin Context API.",
        keywords=["plugins", "extensions", "core", "community", "lifecycle", "api", "hub"],
        html_content="""
<h2>Plugin Architecture & Extensibility</h2>
<p>Theta-IDE features a modular plugin architecture modeled on Obsidian's core/community plugin design.</p>

<h3>Core Plugins vs. Community Plugins</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Aspect</td>
    <td>Core Plugins (e.g. Documentation)</td>
    <td>Community Plugins</td>
  </tr>
  <tr>
    <td><b>Origin</b></td>
    <td>Shipped natively with Theta-IDE source repository (<code>src/plugins/app/bundled/</code>)</td>
    <td>Installed from Theta Hub into user storage (<code>.thetaide/plugins/</code>)</td>
  </tr>
  <tr>
    <td><b>Uninstallable</b></td>
    <td><b>No</b> &mdash; core features cannot be deleted from the filesystem</td>
    <td><b>Yes</b> &mdash; can be uninstalled and removed via trash button</td>
  </tr>
  <tr>
    <td><b>System Impact</b></td>
    <td><b>Zero impact when toggled off</b> &mdash; all panes, listeners, and widgets are cleanly unmounted</td>
    <td>Zero impact when toggled off</td>
  </tr>
  <tr>
    <td><b>Settings Location</b></td>
    <td><i>Settings &rarr; Core Plugins</i></td>
    <td><i>Settings &rarr; Community Plugins</i></td>
  </tr>
</table>

<h3>Plugin Structure</h3>
<p>Every plugin requires a directory containing:</p>
<ol>
  <li><code>plugin.json</code>: Metadata manifest (<code>id</code>, <code>name</code>, <code>description</code>, <code>version</code>, <code>author</code>, <code>icon</code>, <code>entry_point</code>, <code>default_enabled</code>, and <code>core</code> for built-ins). <code>dependencies</code> lists the pip packages it needs; Theta-IDE reports any that are missing instead of failing to load.</li>
  <li><code>__init__.py</code>: Exports a class subclassing <code>Plugin</code>.</li>
</ol>

<h3>The Plugin Context</h3>
<p><code>activate()</code> receives a <code>PluginContext</code>, the plugin's whole interface to the IDE:</p>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Method</td>
    <td>Use</td>
  </tr>
  <tr><td><code>add_sidebar_tab</code> / <code>remove_sidebar_tab</code> / <code>select_sidebar_tab</code></td><td>Contribute a sidebar panel, and show it</td></tr>
  <tr><td><code>register_command</code> / <code>unregister_command</code></td><td>Add a command to the launcher (<code>Ctrl+Space</code>)</td></tr>
  <tr><td><code>contribute_to_slot</code> / <code>remove_from_slot</code></td><td>Put a button or widget into a named UI extension slot</td></tr>
  <tr><td><code>get_setting</code> / <code>set_setting</code></td><td>Read and write values kept for this plugin</td></tr>
  <tr><td><code>spawn_process</code> / <code>track_process</code></td><td>Run a child process that is stopped when the plugin is switched off or the IDE closes</td></tr>
  <tr><td><code>get_current_experiment</code>, <code>get_active_config_path</code>, <code>add_experiment_change_listener</code></td><td>Follow the experiment open in the Experiment tab</td></tr>
  <tr><td><code>open_settings</code>, <code>open_component_hub</code>, <code>log</code>, <code>show_status_message</code></td><td>Open IDE pages and report to the user</td></tr>
  <tr><td><code>get_workspace_dir</code>, <code>get_data_dir</code></td><td>The repository, and the IDE's run-record folder (<code>.thetaide/runs/</code> unless <code>--data-dir</code> moved it)</td></tr>
</table>
<p>Whatever a plugin registers through the context is cleaned up when it is switched off.</p>

<h3>Training Logger Integrations</h3>
<p>Training logger providers run inside the Python process that executes <code>run_pipeline.py</code>, so they use a backend extension point. A provider package registers a callable in the <code>thetaide.run_loggers</code> Python entry-point group. The entry-point name is its config ID; the callable receives its backend settings and a <code>RunLoggerContext</code>, then returns a Lightning logger (or a list of them). Core integrations can also be gated by their plugin setting; disabling the TensorBoard core plugin prevents its provider from starting even when an older experiment still contains <code>tensorboard: true</code>.</p>
<pre><code># pyproject.toml in a logger integration package
[project.entry-points."thetaide.run_loggers"]
wandb = "thetaide_wandb:create_logger"</code></pre>
<p>Configure a provider in an experiment with <code>logging.backends.&lt;id&gt;.enabled</code> and provider-specific keys. For example, a W&amp;B integration can use <code>logging.backends.wandb.project</code> and <code>logging.backends.wandb.entity</code>. Provider code and dependencies must be installed in the environment used to run training. Provider logging settings are separate from optional viewer panes; hiding a pane does not change a run's logger configuration. TensorBoard's existing top-level <code>tensorboard: true</code> setting remains supported for older experiments while the TensorBoard plugin is enabled.</p>

<h3>Plugin Lifecycle Hooks</h3>
<pre><code>from frontend.plugins.base import Plugin, PluginManifest
from frontend.plugins.context import PluginContext

class MyPlugin(Plugin):
    def activate(self, context: PluginContext) -> None:
        # Called when the plugin is enabled
        self.context = context
        self.widget = MyCustomWidget()
        context.add_sidebar_tab(
            tab_id="my_plugin",
            widget=self.widget,
            title="My Extension",
            icon_name="my_icon",
            short_label="Extension"
        )

    def deactivate(self) -> None:
        # Called when toggled off or during IDE shutdown
        # Must clean up all widgets and event listeners
        if self.context:
            self.context.remove_sidebar_tab("my_plugin")
            self.widget.deleteLater()
            self.widget = None
            self.context = None</code></pre>
""",
    ),
    DocArticle(
        id="authoring_plugins",
        title="Authoring Guide: All Component & Plugin Types",
        category="Extensibility",
        summary="Comprehensive developer guide for authoring, registering, and packaging UI Plugins, RL Methods, Models, Environments, and Experiment Recipes.",
        keywords=[
            "authoring", "developer", "plugin", "method", "model", "env", "experiment",
            "component", "hub", "packaging", "protocols", "register_agent", "register_model"
        ],
        html_content="""
<h2>Authoring Guide: All Component & Plugin Types</h2>
<p>Theta-IDE features a modular architecture where nearly every capability &mdash; from UI tabs to RL training algorithms, neural-symbolic models, and benchmark environments &mdash; can be developed, tested, and distributed as an installable component plugin via the <b>Theta Hub</b>.</p>

<h3>Local development through the GUI or terminal</h3>
<p>Hub originals are read-only. Use <b>Edit</b> to inspect them, or <b>Deep Copy</b> to create an independent editable model. Supported copies get a new runtime identity, profiles and parent provenance; shared child components stay shared. Independent copying currently supports directory-based models with safely transformable registrations/imports. <b>Copy YAML</b> creates an editable profile that continues to use shared code.</p>
<pre><code>python run_pipeline.py component inspect blendrl
python run_pipeline.py component copy blendrl my_blendrl
python run_pipeline.py component inspect my_blendrl
python run_pipeline.py component config --help</code></pre>
<p>Use profile paths returned by inspection with <code>component config show</code>, <code>set</code>, <code>copy</code> and <code>validate</code>. To check a configured experiment, use <code>component validate-experiment &lt;group&gt;/&lt;experiment&gt;</code>; it performs preflight and reports effective method model sources without starting training. Static validation is not a forward-pass or shape guarantee.</p>
<p>GUI and terminal share headless component services in <code>src/app/components/</code>. Authored code lives in <code>src/usr/</code>; managed code lives in <code>src/plugins/components/installed</code>; active configs stay in <code>in/config</code>. Retained packages under <code>.thetaide/components/packages/</code> are snapshots; generated context is refreshed under <code>.thetaide/components/context/&lt;id&gt;.md</code>. Managed uninstall retains edits and never removes authored source. Explicit migration provides a dry-run, recovery journal and rollback.</p>

<div style="background-color: rgba(69, 133, 136, 0.12); border-left: 4px solid #83a598; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Component Kinds:</b> Every Theta Hub component declares a <code>kind</code>, which decides where the Hub installs its code and its default config. A component's configs go in a folder named after its package, so experiments refer to them as <code>&lt;package&gt;/&lt;name&gt;</code> (e.g. <code>sb3/ppo</code>).
</div>

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Kind</td>
    <td>Code installs to</td>
    <td>Config</td>
    <td>Registration</td>
  </tr>
  <tr>
    <td><b><code>plugin</code></b></td>
    <td><code>.thetaide/plugins/&lt;id&gt;/</code></td>
    <td>N/A (Managed by settings)</td>
    <td><code>PluginManager</code> scans <code>plugin.json</code></td>
  </tr>
  <tr>
    <td><b><code>method</code></b> (or <code>algorithm</code>)</td>
    <td><code>src/usr/algorithms/&lt;id&gt;/</code></td>
    <td><code>in/config/agent/&lt;package&gt;/</code></td>
    <td><code>@register_algorithm("&lt;prefix&gt;")</code> (alias of <code>register_agent</code>) from <code>src/usr/algorithms/agent_registry.py</code></td>
  </tr>
  <tr>
    <td><b><code>model</code></b></td>
    <td><code>src/usr/models/&lt;id&gt;/</code></td>
    <td><code>in/config/model/&lt;package&gt;/</code></td>
    <td><code>@register_model("&lt;name&gt;")</code> in <code>src/app/core/model_registry.py</code></td>
  </tr>
  <tr>
    <td><b><code>env</code></b></td>
    <td><code>src/usr/environments/&lt;id&gt;/</code></td>
    <td><code>in/config/env/&lt;package&gt;/</code></td>
    <td><code>VectorizedBaseEnv.from_name()</code> factory</td>
  </tr>
  <tr>
    <td><b><code>data</code></b>, <b><code>eval</code></b></td>
    <td><code>src/usr/data/&lt;id&gt;/</code>, <code>src/usr/eval/&lt;id&gt;/</code></td>
    <td><code>in/config/data/&lt;package&gt;/</code>, <code>in/config/eval/</code></td>
    <td>Chosen in an experiment's <b>Data Loaders</b> and <b>Evaluators</b> boxes (<code>data_module:</code>, <code>evaluator:</code>)</td>
  </tr>
  <tr>
    <td><b><code>plot</code></b></td>
    <td><code>src/usr/plotters/&lt;id&gt;/</code></td>
    <td><code>in/config/plot/&lt;package&gt;/</code></td>
    <td>A <code>BasePlotter</code> subclass, discovered automatically; requested under <code>plots:</code></td>
  </tr>
  <tr>
    <td><b><code>experiment</code></b></td>
    <td><code>.thetaide/experiments/&lt;id&gt;/</code></td>
    <td>Its recipes under <code>in/config/experiment/</code></td>
    <td>Dispatched via <code>run_pipeline.py</code></td>
  </tr>
  <tr>
    <td><b><code>suite</code></b></td>
    <td><code>.thetaide/suites/&lt;id&gt;/</code></td>
    <td>&mdash;</td>
    <td>Installs a curated list of other components at once</td>
  </tr>
</table>
<p><b>File &rarr; Scaffold New Component Plugin&hellip;</b> (<code>Ctrl+Shift+P</code>) generates a working starting point for an algorithm, model, environment, data module, evaluator or plotter: its code, its default config and its Hub manifest.</p>

<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;" />

<h3>1. Authoring UI / Frontend Plugins (<code>kind: "plugin"</code>)</h3>
<p>UI plugins extend the Theta-IDE graphical interface by adding custom sidebar workstation tabs, status bar telemetry, or background tools.</p>

<h4>Directory Structure</h4>
<pre><code>my_tool/
├── plugin.json       # Manifest metadata
└── __init__.py       # Plugin class entry point</code></pre>

<h4>Manifest (<code>plugin.json</code>)</h4>
<pre><code>{
  "id": "my_tool",
  "name": "My Custom Tool",
  "version": "1.0.0",
  "description": "Interactive analysis tool for reinforcement learning checkpoints.",
  "author": "Your Name",
  "default_enabled": false,
  "icon": "terminal",
  "kind": "plugin",
  "entry_point": "MyToolPlugin"
}</code></pre>

<h4>Python Implementation (<code>__init__.py</code>)</h4>
<pre><code>from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from frontend.plugins.base import Plugin, PluginManifest
from frontend.plugins.context import PluginContext

class MyToolPlugin(Plugin):
    def activate(self, context: PluginContext) -> None:
        self.context = context

        # Build your custom PyQt6 widget
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.addWidget(QLabel("Welcome to My Custom Tool"))

        # Mount your widget into the IDE sidebar
        context.add_sidebar_tab(
            tab_id="my_tool",
            widget=self.widget,
            title="Custom Tool",
            icon_name="terminal",
            short_label="Tool"
        )

        # Read or write persistent plugin-scoped data
        saved_counter = context.get_setting("click_count", 0)
        context.set_setting("click_count", saved_counter + 1)

    def deactivate(self) -> None:
        # Crucial: Unmount UI and delete widget when toggled off
        if self.context:
            self.context.remove_sidebar_tab("my_tool")
            self.widget.deleteLater()
            self.widget = None
            self.context = None</code></pre>

<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;" />

<h3>2. Authoring RL Method Plugins (<code>kind: "method"</code>)</h3>
<p>Method plugins contribute algorithms and their configuration profiles. A plugin can use the built-in PyTorch Lightning driver or declare a separate runner entrypoint when its framework needs one.</p>

<h4>Directory Structure</h4>
<pre><code>my_method/
├── __init__.py       # REQUIRED: Exposes the agent and triggers @register_agent
├── agent.py          # PyTorch Lightning module implementation
├── my_method.yaml    # Default Tier 1 hyperparameters (deployed to in/config/agent/)
└── plugin.json       # Optional component metadata</code></pre>

<div style="background-color: rgba(254, 128, 25, 0.12); border-left: 4px solid #fe8019; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Crucial Rule:</b> The directory <i>must</i> contain an <code>__init__.py</code> file. Declare package-relative entry points in the manifest. Legacy decorators are inspected statically and loaded through the shared provider catalog.
</div>

<h4>Python Implementation (<code>agent.py</code>)</h4>
<pre><code>import torch
from src.usr.algorithms.base_agent import OfflineAgentBase  # or OnlineAgentBase
from src.usr.algorithms.agent_registry import register_algorithm, register_agent

@register_algorithm("my_algo", "my_algo_variant")
class MyCustomAlgorithm(OfflineAgentBase):
    def __init__(self, obs_dim, n_actions, cfg=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.cql_alpha = getattr(cfg, "cql_alpha", 5.0)
        # Initialize policy, critics, and loss criteria...

    def training_step(self, batch, batch_idx):
        obs, action, reward, next_obs, done = batch
        # Compute Bellman loss and conservative penalty
        loss = self.compute_loss(obs, action, reward, next_obs, done)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 3e-4))</code></pre>

<h4>Export in <code>__init__.py</code></h4>
<pre><code>from .agent import MyCustomAlgorithm

__all__ = ["MyCustomAlgorithm"]</code></pre>

<h4>Default Configuration (<code>my_algo.yaml</code>)</h4>
<p>Deployed automatically to <code>in/config/agent/my_algo.yaml</code>:</p>
<pre><code># @package _global_
agent:
  name: my_algo
  lr: 3e-4
  cql_alpha: 5.0
  batch_size: 256
  gamma: 0.99</code></pre>

<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;" />

<h3>3. Authoring Model Architecture Plugins (<code>kind: "model"</code>)</h3>
<p>A model plugin is a registered PyTorch model. It can be a single network or a parent that constructs child models itself. The framework resolves the parent model name and passes its other configuration values to the registered constructor; nested values remain the model author's responsibility. See <a href="blendrl_hybrid">BlendRL: Authoring a Composite Model Plugin</a> for a complete composition example.</p>

<h4>Directory Structure</h4>
<pre><code>src/usr/models/my_model/
├── __init__.py
├── my_model_model.py       # Defines and registers the nn.Module
├── .theta_component.json  # Hub metadata and parameter schema
└── pyproject.toml          # theta.models entry point
in/config/model/my_model.yaml</code></pre>
<p><b>File &rarr; Scaffold New Component Plugin&hellip;</b> (<code>Ctrl+Shift+P</code>) generates this starting structure for a model. Its package import must load the module containing the registration decorator.</p>

<h4>Implementing a Model</h4>
<p>Register a class with <code>@register_model</code>. The model factory supplies inferred <code>obs_dim</code>, <code>n_actions</code>, <code>device</code>, and <code>env</code> when available, then forwards configured model parameters:</p>
<pre><code>import torch
import torch.nn as nn
from src.app.core.model_registry import register_model

@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2, hidden_sizes=None, **kwargs):
        super().__init__()
        hidden_sizes = hidden_sizes or [64, 64]
        dims = [obs_dim, *hidden_sizes, n_actions]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, obs, **kwargs):
        return self.layers(obs.float().reshape(obs.shape[0], -1))</code></pre>
<p>Keep the constructor tolerant of framework-provided keyword arguments, and return the output shape expected by the selected training algorithm. Put defaults in <code>in/config/model/&lt;name&gt;.yaml</code>; expose editable nested parameters through <code>parameter_schema</code> in the component manifest.</p>

<h4>Optional Runtime Hooks</h4>
<p>Implement an interface from <code>src.app.core.protocols</code> only when the corresponding algorithm or runner consumes it. Available interfaces include:</p>
<ul>
  <li><b><code>AuxiliaryLossProtocol</code></b> for named auxiliary losses.</li>
  <li><b><code>OptimizerGroupsProtocol</code></b> for custom optimizer parameter groups.</li>
  <li><b><code>DynamicTopologyProtocol</code></b> for models whose parameter topology changes during training.</li>
  <li><b><code>ExtraStateProtocol</code></b> for checkpoint state that is not part of the model weights.</li>
  <li><b><code>HasModelCallbacks</code></b> for models that provide Lightning callbacks.</li>
</ul>

<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;" />

<h3>4. Authoring Environment Plugins (<code>kind: "env"</code>)</h3>
<p>Environment plugins package simulator definitions, reward shaping wrappers, and domain valuation logic.</p>

<h4>Directory Structure</h4>
<pre><code>my_custom_env/
├── __init__.py       # Registration or simulator hooks
├── env.py            # Environment wrapper or vectorization
├── reward.py         # Potential-based reward shaping functions
└── my_custom_env.yaml# Deployed to in/config/env/my_custom_env.yaml</code></pre>

<h4>Declarative Environment Configuration</h4>
<p>Every environment declares its operational properties declaratively in its YAML:</p>
<pre><code># @package _global_
env:
  name: my_custom_env
  monitor_metric: "eval/reward" # Target metric for early stopping and tuning
  preprocess_on_load: false
  obs_dim: 8
  n_actions: 4</code></pre>

<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;" />

<h3>5. Authoring Experiment Recipe Plugins (<code>kind: "experiment"</code>)</h3>
<p>Experiment recipes tie environments, agents, paradigms, and cluster resources into reproducible benchmarks.</p>
<pre><code># in/config/experiment/benchmark/_base.yaml
# @package _global_
defaults:
  - override /env: gym_classic_control/cartpole
paradigm: online_rl

# in/config/experiment/benchmark/my_experiment.yaml
# @package _global_
methods:
  ppo_dnn:
    agent: sb3/ppo
    model: basics/dnn
  a2c_dnn:
    agent: sb3/a2c
    model: basics/dnn

resources:
  time: "02:00:00"
  gpus: 1
  cores: 8
  memory: "16G"

total_timesteps: 100000
eval_episodes: 20
num_checkpoints: 10              # or checkpoint_every_steps: 10000 (one or the other)</code></pre>
<p><b>Checkpoints</b> (online RL): set <code>checkpoint_every_steps</code> to save a checkpoint each time training passes another N timesteps, or <code>num_checkpoints</code> to save that many, spread evenly over <code>total_timesteps</code>. The last one always falls at the end of the run. They are saved as <code>step_&lt;timesteps&gt;.ckpt</code> (<code>.zip</code> for Stable-Baselines3) next to <code>best_model</code>. Setting both is an error. In the experiment editor, <b>+ &rarr; Checkpoints</b> adds the row; its menu switches between the two.</p>
<p><b>Recorded metrics</b>: list the metrics a run records, by name or by whole section, and the work behind the others is skipped, so the run is faster:</p>
<pre><code>telemetry:
  metrics: [eval/reward, losses/]   # a name, or a section ending in /</code></pre>
<p>Only the listed metrics reach <code>metrics.csv</code>, the live feed and TensorBoard; step, epoch and transitions are always kept. With no <code>eval/</code> entry, no evaluation episodes run (and no best-by-reward checkpoint is kept; the final model is still saved); with no <code>system/</code> entry, CPU and memory are not sampled. Leaving the list out records everything. A sweep must keep the metric it optimises. In the experiment editor, <b>+ &rarr; Recorded Metrics</b> adds a box of checkboxes: a section's own box records all of it, and the metrics under it are the ones this experiment's newest earlier run logged.</p>

<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;" />

<h3>6. Packaging & Publishing to the Theta Hub</h3>

<h4>1-Click Experiment-to-Plugin Export</h4>
<p>Theta-IDE provides export of an experiment and its recognized configuration/source/asset dependencies:</p>
<ul>
  <li><b>One experiment:</b> Click <b>Export</b> in the experiment tree's toolbar (or <code>Ctrl+Shift+S</code>) &rarr; choose <i>Theta Hub Component Archive (.zip)</i>. The IDE automatically traces the recipe YAML, base configs, environment wrappers, domain rules, custom model architectures, and optional checkpoint weights (<i>Include trained checkpoint weights</i>).</li>
  <li><b>A whole group:</b> right-click the group in the tree &rarr; <b>Export Experiment Plugin&hellip;</b>.</li>
  <li><b>CLI Command:</b>
    <pre><code>python run_pipeline.py export &lt;group&gt;/&lt;experiment&gt;
# With trained checkpoints:
python run_pipeline.py export &lt;group&gt;/&lt;experiment&gt; --include-weights --out results/exports/my_exp.zip</code></pre>
  </li>
</ul>

<h4>Universal Python Packaging (<code>pyproject.toml</code>)</h4>
<p>Some components can also be packaged as a standard Python package and installed via <code>pip</code>, declaring entry points:</p>
<pre><code>[project.entry-points."theta.models"]
my_model = "my_pkg.models:MyModel"

[project.entry-points."theta.agents"]
my_agent = "my_pkg.agents:MyAgent"

[project.entry-points."theta.datamodules"]
my_dm = "my_pkg.data:MyDataModule"

[project.entry-points."theta.components"]
my_component = "my_pkg.components:MyRunnerOrCallback"</code></pre>
<p><code>theta.datamodules</code> and <code>theta.components</code> are registered during paradigm component discovery. <code>theta.environments</code> is resolved by the environment factory, and <code>theta.tuners</code> by tuning-component selection. <code>theta.models</code> and <code>theta.agents</code> require explicit <code>load_all_plugins()</code> or <i>File &rarr; Reload Component Code</i> (<code>Ctrl+Alt+R</code>); fresh training processes do not call that loader automatically. Evaluator and GUI-plugin entry-point names are recognized by the helper but have no automatic activation adapter. GUI plugins use <code>plugin.json</code>. Entry points alone do not create workspace YAML profiles; Hub packaging remains the complete install/configuration route.</p>

<h4>Manual Release Archive & Hub Registry (<code>dist/index.json</code>)</h4>
<p>The default Theta Hub feed is <code>https://raw.githubusercontent.com/ThetaIDE/ThetaHub/main/dist/index.json</code>; the client also supports fallback feeds and caches responses in <code>.thetaide/cache/hub/</code>. Check <code>DEFAULT_REGISTRY_URL</code> in <code>frontend/hub/client.py</code> and follow <code>docs/development/THETA_HUB_RELEASE_WORKFLOW.md</code> for production releases. For a standalone release archive, package the directory and compute its SHA-256:</p>
<pre><code>zip -r my_cql-1.0.0.zip my_method/
shasum -a 256 my_cql-1.0.0.zip</code></pre>
<p>Publishing requires a versioned archive, matching registry metadata and feeds, and client verification. Exporting a ZIP does not publish it. The GUI currently has no local-archive drag-and-drop/import action.</p>
""",
    ),
    DocArticle(
        id="cli_workflows",
        title="Command-Line (CLI) Workflows",
        category="Workflow & Tools",
        summary="CLI reference for run_pipeline.py: running experiments, overrides, plotting, sweeps, and cluster submission.",
        keywords=["cli", "terminal", "commands", "run_pipeline", "plot", "optuna", "sweeps", "overrides"],
        html_content="""
<h2>CLI & Command Reference</h2>
<p>The experiment pipeline and component-development operations below are available from the terminal. Run them from the repository root in Theta's Python environment, after installing the required components. Some desktop actions, including Hub browsing and system Trash, have no terminal subcommand.</p>

<h3>Running Experiments</h3>
<p>Experiments are named <code>&lt;group&gt;/&lt;experiment&gt;</code>, matching their path under <code>in/config/experiment/</code> (an experiment in a subfolder is <code>&lt;group&gt;/&lt;subfolder&gt;/&lt;experiment&gt;</code>):</p>
<pre><code># Train an experiment
python run_pipeline.py &lt;group&gt;/&lt;experiment&gt;

# Override any config value with Hydra key=value syntax
python run_pipeline.py &lt;group&gt;/&lt;experiment&gt; total_timesteps=50000 eval_episodes=10

# Check that a config composes and validates, without training
python run_pipeline.py &lt;group&gt;/&lt;experiment&gt; dry_run=true

# Replace with an experiment supplied by your installed components
python run_pipeline.py <group>/<experiment></code></pre>
<p>Each command-line run writes its results under the recipe's <code>experiment_id</code>, replacing the previous run with that ID; pass <code>experiment_id=&lt;new name&gt;</code> to keep both. In the IDE, enable Timestamp experiment to give reruns a timestamped ID and keep previous results. Run <code>python run_pipeline.py --help</code> for the full list of orchestration overrides.</p>

<h3>Common Overrides</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Override</td>
    <td>Effect</td>
  </tr>
  <tr><td><code>dry_run=true</code></td><td>Validate the config and exit</td></tr>
  <tr><td><code>plot_only=true</code></td><td>Run only the plotting phase</td></tr>
  <tr><td><code>no_plot=false</code></td><td>Enable plotting of configured figures and tables (disabled by default)</td></tr>
  <tr><td><code>no_online=true</code> / <code>no_offline=true</code></td><td>Skip the online or offline training phase</td></tr>
  <tr><td><code>experiment_id=&lt;name&gt;</code></td><td>Override the results directory name</td></tr>
  <tr><td><code>site=ncshare</code> / <code>site=arc</code> (and <code>ncshare_gpu</code>, <code>ncshare_common</code>)</td><td>Submit to a Slurm cluster instead of running locally (see <a href="cluster_slurm">Cluster Execution</a>)</td></tr>
  <tr><td><code>consolidate=true</code></td><td>Put every method into one Slurm GPU job</td></tr>
  <tr><td><code>remake=true</code></td><td>Recalculate summaries</td></tr>
</table>

<h3>Plotting</h3>
<p>Select plotters in the experiment’s <b>Figures and Tables</b> box or its <code>plots</code> configuration. If no plotters are selected, plotting is skipped.</p>
<pre><code># Re-run the plotting phase for an experiment
python run_pipeline.py &lt;group&gt;/&lt;experiment&gt; plot_only=true

# Or call the plot manager directly, with the run's ID (its folder under results/logs/&lt;group&gt;/)
python plot/manager.py &lt;group&gt;/&lt;experiment_id&gt;

# Start from an empty plot directory
python plot/manager.py &lt;group&gt;/&lt;experiment_id&gt; --wipe</code></pre>

<h3>Exporting</h3>
<pre><code># An experiment and everything it depends on, as a Theta Hub archive
python run_pipeline.py export &lt;group&gt;/&lt;experiment&gt; --include-weights --out results/exports/my_exp.zip</code></pre>

<h3>Optuna Hyperparameter Sweeps</h3>
<p>Install the optional tuner from the project directory with <code>pip install -e ".[optuna]"</code>.</p>
<p>Add a Hyperparameter Sweep block in the experiment editor, choose an objective metric and direction, and mark the method parameters to search. The Enabled switch preserves tuning settings when off; methods with search spaces cannot run until tuning is enabled or their search spaces are removed. Each tuning trial writes its own checkpoint, and the winning checkpoint and parameters are saved in the method's checkpoint folder.</p>
<p>The Optuna tuner offers TPE, random, or scrambled Sobol QMC sampling, a sampler seed, trial count, batch size and storage. QMC samples categorical parameters independently. A batch runs concurrently only with a parallel Hydra launcher. Select <b>new</b> to create a study or <b>resume</b> and enter its exact name to add more trials. For multiple methods, include <code>{method}</code> in a custom study name so each method has its own study.</p>
<p>Median early stopping reports the selected objective after each Lightning validation epoch and can prune underperforming trials. Configure how many trials finish first and how many validation epochs to wait. The objective must be logged during validation; SB3 pruning and multi-process Lightning are not yet supported. A time budget prevents new batches after it expires, while allowing an active batch to finish.</p>
<p>Search dimensions support scalar choices, integer and numeric ranges, optional numeric steps, and positive log-scaled numeric ranges.</p>
<p>The failed-trial allowance controls how many trial failures a sweep can continue past in the current run. Pruned trials do not count as failures.</p>
<p>Add the <code>tuning</code> figure in Figures and Tables to generate a trial table and objective-history plot from the saved tuning summary.</p>
<p>Optuna studies created before trial identities were recorded can still be inspected in the dashboard. Their winning checkpoints cannot be promoted automatically without an explicit trial-to-checkpoint mapping.</p>
<pre><code>python run_pipeline.py &lt;group&gt;/&lt;tuning_experiment&gt; sweep=true

# Watch the sweep in the Optuna dashboard
python run_pipeline.py &lt;group&gt;/&lt;tuning_experiment&gt; sweep=true dash=true

# Only open the dashboard for an earlier sweep
python run_pipeline.py &lt;group&gt;/&lt;tuning_experiment&gt; dash_only=true</code></pre>
""",
    ),
    DocArticle(
        id="results_browser",
        title="Results Browser",
        category="Interface",
        summary="Every metric, value and file your runs produced: chart, inspect, compare, summarize and export them.",
        keywords=["results", "runs", "metrics", "curves", "compare", "summary", "seeds", "versions", "export",
                  "inspector", "tables", "figures", "reports", "config", "files", "live"],
        html_content=r"""
<h2>Results Browser</h2>
<p>The <a href="ide://results">Results</a> pane (<code>Action + 4</code>) shows everything your runs produced, from every logged value to every file, whichever engine trained them and however they were launched (Theta-IDE, the command line, or Slurm). It reads <code>results/</code> directly; the refresh button re-reads it.</p>

<h3>Choosing Runs</h3>
<p>The tree on the left lists every run in <code>results/logs</code>: group &rarr; experiment &rarr; agent. Tick a group, an experiment or an agent; ticking a parent ticks everything under it. When an agent has several <b>versions</b> (repeated runs, e.g. seeds, stored as <code>version_N</code> folders) they appear as children, with the newest ticked. Use the filter box to find runs by name.</p>

<h3>Curves</h3>
<ul>
  <li><b>One chart per metric</b> the ticked runs logged, grouped by prefix (<code>eval/</code>, <code>losses/</code>, <code>val/</code>, <code>time/</code>, …). Nothing is whitelisted: every numeric column of every <code>metrics.csv</code> gets a chart, including metrics from Stable-Baselines3 or custom engines.</li>
  <li><b>Hide</b> a chart with its button, or use <b>Plots ▾</b> to show or hide any metric. Hidden plots are remembered under <code>[results] hidden_plots</code> in <code>settings.toml</code>.</li>
  <li><b>X-axis:</b> environment steps, logged step, epoch, wall time or training time, whichever the runs logged. <b>Smoothing</b> applies an exponential moving average to the plotted curves.</li>
  <li><b>Versions:</b> <i>Separate</i> draws each version; <i>Mean ± std</i> or <i>Mean ± SEM</i> combine an agent's ticked versions into one curve with a shaded band.</li>
  <li><b>Primary:</b> the series whose logged steps hovering snaps to. It is drawn on top and thicker, and listed first in the inspector. <i>Nearest</i> snaps to whichever series is closest to the mouse.</li>
  <li><b>Point inspector:</b> hover over any chart to see every metric of every ticked run at that step. Runs that did not log exactly that step show their nearest logged point.</li>
  <li><b>Live:</b> while the pane is open, ticked runs that are still training update every few seconds with their new values.</li>
</ul>

<h3>Summary</h3>
<p>One sortable row per series and metric: final, best (the highest for rewards, the lowest for losses; blank when a metric has no better direction), where the best was, minimum, maximum, mean, area under the curve, and the number of points. Pick one metric and enter a <b>threshold</b> to see when each run first reached it.</p>

<h3>Tables, Figures, Reports, Config and Files</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Tab</td>
    <td>Shows</td>
  </tr>
  <tr><td><b>Tables</b></td><td>Every CSV a run wrote, including each raw <code>metrics.csv</code>: every row and column, sortable, with a plot of any column against another</td></tr>
  <tr><td><b>Figures</b></td><td>Every plot image; scroll to zoom, drag to pan</td></tr>
  <tr><td><b>Reports</b></td><td>Markdown and text reports, rendered</td></tr>
  <tr><td><b>Config</b></td><td>Every config and metadata file, plus a table of the settings that differ between the ticked runs. <b>Load in experiment builder</b> puts a run's builder settings into the Experiment tab (for runs Theta-IDE launched, or single-method CartPole/PPO runs).</td></tr>
  <tr><td><b>Files</b></td><td>Every file of the ticked runs, checkpoints and TensorBoard logs included. Files too large or of a kind that cannot be previewed can still be saved with <b>Save a copy…</b></td></tr>
</table>

<h3>Export</h3>
<p><b>Export ▾</b> writes the visible curves as CSV (one row per point, as drawn), the Summary as CSV, or the charts as a PNG image.</p>

<p>The tabs along the top are <b>Curves</b>, <b>Summary</b>, <b>Tables</b>, <b>Figures</b>, <b>Reports</b>, <b>Config</b> and <b>Files</b>. To follow the run training now, use the <a href="ide://monitor">Training Monitor</a>; to plot a few runs together quickly, the Plot Viewer's run list.</p>
""",
    ),
    DocArticle(
        id="tensorboard_tab",
        title="TensorBoard Tab",
        category="Interface",
        summary="View TensorBoard dashboards inside the IDE for runs launched with Log to TensorBoard.",
        keywords=["tensorboard", "logs", "dashboard", "scalars", "webengine", "browser"],
        html_content=r"""
<h2>TensorBoard Tab</h2>
<p>Runs with <b>Log to TensorBoard</b> turned on write event logs to <code>results/tensorboard/</code>. Add the option from the <b>+</b> beside the Experiment heading or set <code>tensorboard: true</code> in YAML. The <a href="ide://tensorboard">TensorBoard</a> tab runs a server over that folder. It is hidden by default; open it with <code>Action + 6</code> or <i>View &rarr; TensorBoard</i>. It comes from the TensorBoard core plugin in <i>Settings &rarr; Core Plugins</i>. Turning that plugin off removes its pane and Experiment control, and disables TensorBoard logging for runs launched by Theta-IDE.</p>

<h3>How It Works</h3>
<ul>
  <li><b>Start server</b> starts a local TensorBoard server as a child process of Theta-IDE, bound to <code>127.0.0.1</code> on a free port. The status line shows <i>Starting&hellip;</i>, then <i>Running</i> and its address.</li>
  <li><b>Open in browser</b> opens the dashboard in your web browser.</li>
  <li><b>Stop server</b> shuts it down. It also stops when you switch the plugin off or quit Theta-IDE.</li>
</ul>

<h3>Requirements</h3>
<ul>
  <li>TensorBoard is an optional integration. Install <code>theta-ide[tensorboard]</code> in the training environment to enable its run logger; if the viewer dependency is missing, the plugin reports that instead of starting.</li>
</ul>

<div style="background-color: rgba(254, 128, 25, 0.12); border-left: 4px solid #fe8019; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Force-quitting Theta-IDE</b> (killing its process) can leave the TensorBoard server running; see <a href="troubleshooting">Troubleshooting &amp; FAQ</a> to stop it.
</div>
""",
    ),
    DocArticle(
        id="console_pane",
        title="Console",
        category="Interface",
        summary="The Console streams pipeline output and IDE messages, and accepts a few quick commands.",
        keywords=["console", "log", "output", "commands", "status", "help"],
        html_content=r"""
<h2>Console</h2>
<p>The <a href="ide://console">Console</a> pane is the IDE's log. It streams the training pipeline's output while a run is active, along with IDE messages such as saves, queue changes, plugin activity and the pipeline's status.</p>
<p>It is not a shell. For an interactive shell, use the <a href="ide://terminal">Terminal</a> tab.</p>

<h3>Commands</h3>
<p>Type a command into the input line at the bottom and press Enter:</p>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Command</td>
    <td>What it does</td>
  </tr>
  <tr><td><code>help</code></td><td>List the available commands</td></tr>
  <tr><td><code>status</code></td><td>Show the active run (its status, experiment ID and job ID, or <i>idle</i>) and how many run records there are</td></tr>
  <tr><td><code>config</code></td><td>Print the <code>run_pipeline.py</code> command equivalent to the loaded config, to run it outside the IDE</td></tr>
  <tr><td><code>clear</code></td><td>Clear the console</td></tr>
</table>
""",
    ),
    DocArticle(
        id="arxiv_papers",
        title="arXiv Papers",
        category="Interface",
        summary="Find papers about the algorithms, models and environments in your workspace, read abstracts and cite them.",
        keywords=["arxiv", "papers", "literature", "citation", "bibtex", "abstract", "pdf", "tags", "research"],
        html_content=r"""
<h2>arXiv Papers</h2>
<p>The <a href="ide://arxiv_papers">arXiv Papers</a> tab searches <a href="https://arxiv.org">arXiv</a> for the literature behind your components. It comes from a core plugin that is on by default; switch it off in <i>Settings &rarr; Core Plugins</i>.</p>

<h3>Searching</h3>
<ol>
  <li>Pick a component from the <b>Components</b> list (methods, models, environments, experiments and plugins), or pick one or more <b>Tags</b>. Use the filter box to find one.</li>
  <li>Choose <b>Match all tags</b> or <b>Match any tag</b>, and optionally type extra search words.</li>
  <li>Pick an order (<b>Relevance</b>, <b>Newest</b> or <b>Recently updated</b>) and click <b>Search arXiv</b>. The query that will be sent is shown as you build it.</li>
</ol>
<p>Tags come from the components themselves (built-in tags and the Theta Hub's), and <b>Edit tags</b> changes the tags searched for a component. Your edits are kept with the plugin's settings.</p>

<h3>Reading and Citing</h3>
<ul>
  <li>Select a result to read its abstract, authors and dates.</li>
  <li><b>Open on arXiv</b> or <b>Open PDF</b> opens it in your browser; <b>Copy BibTeX</b> copies a citation.</li>
  <li>Results are cached for a day (in <code>arxiv_cache.json</code> in the IDE's data folder), so repeating a search doesn't ask arXiv again. Searches run one at a time, which keeps within arXiv's rate limit.</li>
</ul>
<p>Needs a network connection. Thank you to arXiv for use of its open access interoperability.</p>
""",
    ),
    DocArticle(
        id="settings_guide",
        title="Settings & settings.toml",
        category="Customization",
        summary="Tour of the seven Settings pages, and how settings.toml files store and override your preferences.",
        keywords=["settings", "preferences", "settings.toml", "toml", "backend", "storage", "plugins", "hotkeys", "config"],
        html_content=r"""
<h2>Settings & settings.toml</h2>
<p>Open Settings with <code>Action + 0</code> or <i>View &rarr; Settings</i>.</p>

<h3>Settings Pages</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Page</td>
    <td>What you can change</td>
  </tr>
  <tr><td><b>Appearance</b></td><td>Color theme, display zoom, which sidebar panels are shown, auto-hide sidebar, <b>Experiment Panel Configuration</b> (show resolved defaults, algorithm and model defaults in methods, the visual layer editor), and the Results panel layout</td></tr>
  <tr><td><b>Core Plugins</b></td><td>Turn built-in plugins (Text Editor, Documentation, TensorBoard, arXiv Papers, Visual Model Builder) on or off</td></tr>
  <tr><td><b>Community Plugins</b></td><td>Install, configure, or remove plugins from the Theta Hub</td></tr>
  <tr><td><b>Hotkeys</b></td><td>Action key, terminal precedence, the <code>Action + digit</code> pane mapping, and menu shortcuts</td></tr>
  <tr><td><b>Training Pipeline</b></td><td>The Python environment that composes configs and runs training, with a <b>Check pipeline</b> button</td></tr>
  <tr><td><b>Terminal</b></td><td>The shell the Terminal tab starts, and whether the project venv comes first on PATH</td></tr>
  <tr><td><b>Text Editor</b></td><td>Default text editor for files opened inside Theta, and line wrapping. Enable the Text Editor core plugin for the built-in CodeMirror 6 Editor panel.</td></tr>
  <tr><td><b>Workspace &amp; Storage</b></td><td>Data and database locations, open the settings file, and reset the UI layout</td></tr>
</table>

<h3>Settings Files</h3>
<p>Every setting is stored in a TOML file you can also edit by hand. Two files are read, and the workspace file wins:</p>
<ol>
  <li><code>~/.config/thetaide/settings.toml</code>: your defaults for every workspace (on Windows, <code>C:\Users\&lt;you&gt;\.config\thetaide\settings.toml</code>).</li>
  <li><code>.thetaide/settings.toml</code> in the repository: this workspace. Created with defaults on first launch.</li>
</ol>
<p>Open the workspace file with <i>View &rarr; Preferences: Open Settings File</i>. Changes you save in an editor apply immediately, without restarting.</p>

<h3>Sections</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Section</td>
    <td>Keys</td>
  </tr>
  <tr><td><code>[appearance]</code></td><td><code>theme</code>, <code>zoom</code>, and the experiment panel's <code>show_resolved_defaults</code>, <code>show_method_agent_defaults</code>, <code>show_method_model_defaults</code>, <code>show_all_default_params</code></td></tr>
  <tr><td><code>[sidebar]</code></td><td><code>order</code> and <code>visible</code> pane lists, <code>bins</code>, <code>auto_hide</code></td></tr>
  <tr><td><code>[plugins]</code></td><td><code>enabled</code> and <code>disabled</code> plugin IDs</td></tr>
  <tr><td><code>[components]</code></td><td><code>confirm_delete</code></td></tr>
  <tr><td><code>[hotkeys]</code></td><td><code>enabled</code>, <code>action_key</code>, <code>leader_timeout</code>, <code>terminal_precedence</code>, <code>panes</code></td></tr>
  <tr><td><code>[terminal]</code></td><td><code>shell</code>, <code>cwd</code>, <code>activate_venv</code></td></tr>
  <tr><td><code>[editor]</code></td><td><code>default_text_editor</code> (<code>system</code> or <code>text_editor</code>), <code>wrap_lines</code></td></tr>
  <tr><td><code>[results]</code></td><td><code>hidden_plots</code></td></tr>
  <tr><td><code>[shortcuts]</code></td><td>Menu and training shortcut overrides, e.g. <code>launch_training = "F6"</code></td></tr>
</table>

<p>See also <a href="themes">Themes &amp; Theme Builder</a>, <a href="sidebar_layout">Sidebar &amp; Layout</a>, and <a href="hotkeys">Keyboard Shortcuts</a>.</p>
""",
    ),
    DocArticle(
        id="themes",
        title="Themes & Theme Builder",
        category="Customization",
        summary="Switch between the built-in color themes or design your own with the Theme Builder.",
        keywords=["theme", "themes", "colors", "palette", "dark", "light", "gruvbox", "catppuccin", "nord", "dracula", "theme builder"],
        html_content=r"""
<h2>Themes & Theme Builder</h2>

<h3>Switching Themes</h3>
<p>Pick a theme from <i>View &rarr; Themes</i>, or from <i>Settings &rarr; Appearance &rarr; Active Theme</i>. The change applies immediately and is saved as <code>theme</code> under <code>[appearance]</code> in <code>settings.toml</code>.</p>

<h3>Built-In Themes</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Family</td>
    <td>Themes</td>
  </tr>
  <tr><td>Catppuccin</td><td>Catppuccin (default), Catppuccin Macchiato, Catppuccin Latte</td></tr>
  <tr><td>Gruvbox</td><td>Gruvbox Dark, Gruvbox Light</td></tr>
  <tr><td>Classic</td><td>Nord, Dracula, Paper</td></tr>
  <tr><td>Olympian</td><td>Apollo, Athena, Ares, Dionysus, Poseidon</td></tr>
</table>

<h3>Theme Builder</h3>
<p>Open <i>View &rarr; Theme builder&hellip;</i> to create your own palette:</p>
<ol>
  <li>Under <b>Start from</b>, pick the preset to base your theme on.</li>
  <li>Give it a <b>Theme name</b>.</li>
  <li>Click <b>Choose&hellip;</b> next to any color role to change it. Valid edits preview across the whole app as you make them.</li>
  <li>Click <b>Save &amp; apply</b>. Your theme then appears in the Themes menu alongside the built-in ones.</li>
</ol>
<p><b>Reset to preset</b> discards your edits, and <b>Cancel</b> restores the theme you had before opening the builder.</p>
""",
    ),
    DocArticle(
        id="sidebar_layout",
        title="Sidebar & Layout",
        category="Customization",
        summary="Show or hide panels, reorder sidebar tabs, auto-hide the sidebar, and restore the default layout.",
        keywords=["sidebar", "layout", "panels", "tabs", "hide", "show", "reorder", "auto-hide", "reset", "restore"],
        html_content=r"""
<h2>Sidebar & Layout</h2>

<h3>Showing and Hiding Panels</h3>
<p>To keep the sidebar short, some panels are <b>hidden by default</b>, such as Plots, TensorBoard and Queue. Turn any panel on or off with its switch in <i>Settings &rarr; Appearance &rarr; Sidebar Panels Visibility</i>. At least one panel must stay visible, so the last switch can't be turned off.</p>
<p>A hidden panel is still reachable: choose it from the <i>View</i> menu, or press its <code>Action + digit</code> hotkey.</p>

<h3>Reordering Tabs</h3>
<p>Drag a sidebar tab up or down to move it: drop it on the top or bottom edge of another tab. The new order is saved as <code>order</code> under <code>[sidebar]</code> in <code>settings.toml</code>.</p>

<h3>Grouping Tabs into Bins</h3>
<p>Drop a tab onto the <b>middle</b> of another tab to group the two into a named <b>bin</b>, or right-click a tab and choose <b>Add to new bin</b>. Drop more tabs onto the bin's header to add them, and drag a tab out of the bin to take it out again.</p>
<ul>
  <li>Click a bin's header to collapse or expand it; double-click it to rename it.</li>
  <li>Right-click the header for <b>Rename</b>, <b>Collapse</b> / <b>Expand</b> and <b>Ungroup</b>.</li>
  <li>Bins are saved as <code>bins</code> under <code>[sidebar]</code>.</li>
</ul>

<h3>Auto-Hide Sidebar</h3>
<p>Turn on <i>View &rarr; Auto-hide sidebar</i> (or the switch in <i>Settings &rarr; Appearance</i>) to give the panels the full window width. Hover over the left edge of the window to bring the sidebar back.</p>

<h3>Restoring Defaults</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>To reset</td>
    <td>Use</td>
  </tr>
  <tr><td>Tab order and visibility</td><td><i>Settings &rarr; Appearance &rarr; Restore default sidebar</i> (the original order, with the default panels shown)</td></tr>
  <tr><td>Pane sizes and docking</td><td><i>View &rarr; Restore default layout</i>, or <i>Settings &rarr; Workspace &amp; Storage &rarr; Reset UI layout</i></td></tr>
</table>
""",
    ),
    DocArticle(
        id="troubleshooting",
        title="Troubleshooting & FAQ",
        category="Getting Started",
        summary="Answers to common setup, pipeline, OpenGL, and cluster connection questions.",
        keywords=["troubleshooting", "faq", "errors", "pipeline", "environment", "opengl", "debug"],
        html_content="""
<h2>Troubleshooting & FAQ</h2>

<h3>1. Status bar says "Pipeline unavailable"</h3>
<p>Theta-IDE imports the training pipeline directly, so it must run from the project venv:</p>
<ul>
  <li>Activate <code>venv/</code> and run <code>pip install -e ".[gui]"</code>, then start the app with <code>python -m frontend</code> from the repository root.</li>
  <li><i>Settings &rarr; Training Pipeline</i> shows which Python the app is using and has a <b>Check pipeline</b> button.</li>
  <li>Simulated demo runs (<b>Start simulated demo</b> in the Queue tab (<code>Ctrl+F5</code>)) work without the training environment.</li>
</ul>

<h3>2. A run fails because a component or environment is missing</h3>
<p>Algorithms, models and environments are supplied by components. Install from the Theta Hub the components named by the experiment:</p>
<ul>
  <li><code>references missing component(s): &hellip; Install them via the Component Hub</code>: install the listed components from the Hub.</li>
  <li>A required environment is missing: install the environment component and select its registered configuration in the experiment.</li>
  <li>Check any experiment without training with <code>python run_pipeline.py &lt;group&gt;/&lt;experiment&gt; dry_run=true</code>.</li>
</ul>

<h3>3. OpenGL / Qt Display Errors on Headless or Cluster Nodes</h3>
<p>When running tests or GUI components on a headless server, export the offscreen platform plugin:</p>
<pre><code>export QT_QPA_PLATFORM=offscreen</code></pre>

<h3>4. Where are my training results and logs stored?</h3>
<p>Results adhere to a standardized hierarchical structure:</p>
<ul>
  <li><b>Metrics & Logs:</b> <code>results/logs/[GROUP]/[EXP_ID]/[AGENT]/version_X/metrics.csv</code></li>
  <li><b>Checkpoints:</b> <code>results/checkpoints/[GROUP]/[EXP_ID]/[AGENT]/</code></li>
  <li><b>Replay Buffers:</b> <code>results/datasets/[GROUP]/[EXP_ID]/[AGENT]/</code></li>
  <li><b>Plots & Reports:</b> <code>results/plots/[GROUP]/[EXP_ID]/</code></li>
</ul>

<h3>5. How do I reset the IDE's layout or cached settings?</h3>
<ul>
  <li><b>Sidebar tabs:</b> <i>Settings &rarr; Appearance &rarr; Restore default sidebar</i> restores the original tab order and shows the default panels.</li>
  <li><b>Pane sizes and docking:</b> <i>Settings &rarr; Workspace &amp; Storage &rarr; Reset UI layout</i>, or <i>View &rarr; Restore default layout</i>.</li>
  <li><b>Everything else:</b> edit or delete <code>.thetaide/settings.toml</code>. Theta-IDE recreates it with defaults. See <a href="settings_guide">Settings &amp; settings.toml</a>.</li>
</ul>

<h3>6. Windows: <code>Activate.ps1 cannot be loaded because running scripts is disabled</code></h3>
<p>PowerShell blocks the virtual environment's activation script by default. Allow it once for your account:</p>
<pre><code>Set-ExecutionPolicy -Scope CurrentUser RemoteSigned</code></pre>

<h3>7. Windows: <code>Python was not found; run without arguments to install from the Microsoft Store</code></h3>
<p>Windows is using its Store shortcut instead of your Python install. Turn off the <code>python.exe</code> and <code>python3.exe</code> entries in <i>Settings &rarr; Apps &rarr; Advanced app settings &rarr; App execution aliases</i>, or activate the virtual environment first.</p>

<h3>8. The Terminal tab is blank or says no shell was found (Windows)</h3>
<p>The Windows terminal needs <code>pywinpty</code>, which the <code>gui</code> extra installs. Reinstall with <code>pip install -e ".[gui]"</code> in the project venv. To use a particular shell, pick it in <i>Settings &rarr; Terminal</i> or set <code>shell</code> under <code>[terminal]</code> in <code>settings.toml</code>, for example <code>shell = "pwsh -NoLogo"</code>.</p>

<h3>9. Linux: <code>libEGL.so.1: cannot open shared object file</code></h3>
<p>Qt needs system graphics libraries that minimal Linux installs and CI runners often lack. On Ubuntu 24.04:</p>
<pre><code>sudo apt-get install libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3 libnss3 libxcomposite1 \
    libxdamage1 libxrandr2 libxtst6 libxkbfile1 libgbm1 libasound2t64</code></pre>

<h3>10. TensorBoard keeps running after Theta-IDE was force-closed</h3>
<p>Quitting normally, or <b>Stop server</b> in the TensorBoard tab, shuts the server down. If Theta-IDE was killed another way, find the leftover process. It runs as <code>python -m tensorboard.main</code>:</p>
<pre><code># macOS / Linux
pkill -f tensorboard.main

# Windows PowerShell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object CommandLine -like "*tensorboard.main*" |
    ForEach-Object { Stop-Process -Id $_.ProcessId }</code></pre>
""",
    ),
]


def get_all_articles() -> List[DocArticle]:
    """Return all available documentation articles."""
    return list(ARTICLES)


def get_article_by_id(article_id: str) -> DocArticle | None:
    """Retrieve an article by its unique identifier."""
    for art in ARTICLES:
        if art.id == article_id:
            return art
    return None


def search_articles(query: str) -> List[DocArticle]:
    """Filter articles by search query against title, summary, keywords, and content."""
    query = query.strip().lower()
    if not query:
        return get_all_articles()

    results: List[DocArticle] = []
    for art in ARTICLES:
        if (
            query in art.title.lower()
            or query in art.summary.lower()
            or query in art.category.lower()
            or any(query in kw.lower() for kw in art.keywords)
            or query in art.html_content.lower()
        ):
            results.append(art)
    return results
