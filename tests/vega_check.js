// Compile and run Vega-Lite specs with the vendored builds, headless (tests/vega_support.py).
// stdin: JSON list of {spec, signals}; stdout: JSON list of {ok, error, warnings, svg}.
const vega = require("vega");
const vl = require("vega-lite");

async function check({spec, signals, config}) {
  const warnings = [];
  const logger = {
    level() { return this; }, info() { return this; }, debug() { return this; },
    warn(...args) { warnings.push(args.join(" ")); return this; },
    error(...args) { warnings.push("ERROR " + args.join(" ")); return this; },
  };
  try {
    const compiled = vl.compile(spec, {logger, config: config || undefined}).spec;
    const view = new vega.View(vega.parse(compiled), {renderer: "none", logger});
    // "container" sizes need a DOM; give the headless view a fixed size instead.
    if (spec.width === "container") view.width(480);
    if (spec.height === "container") view.height(300);
    await view.runAsync();
    for (const [name, value] of Object.entries(signals || {})) view.signal(name, value);
    await view.runAsync();
    return {ok: true, warnings, svg: await view.toSVG()};
  } catch (err) {
    return {ok: false, error: String(err && err.message || err), warnings};
  }
}

let input = "";
process.stdin.on("data", chunk => { input += chunk; });
process.stdin.on("end", async () => {
  const results = [];
  for (const item of JSON.parse(input)) results.push(await check(item));
  process.stdout.write(JSON.stringify(results));
});
