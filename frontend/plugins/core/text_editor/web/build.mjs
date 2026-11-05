import {build} from "esbuild";
import {readFile, writeFile, readdir} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath} from "node:url";

process.chdir(path.dirname(fileURLToPath(import.meta.url)));
const result = await build({
  entryPoints: ["editor.js"],
  bundle: true,
  minify: true,
  format: "iife",
  target: "chrome112",
  outfile: "vendor/editor.bundle.js",
  legalComments: "linked",
  metafile: true,
});

const packages = new Set();
for (const source of Object.keys(result.metafile.inputs)) {
  const match = source.match(/^node_modules\/((?:@[^/]+\/)?[^/]+)/);
  if (match) packages.add(match[1]);
}
const notices = ["THETA TEXT EDITOR — THIRD-PARTY NOTICES\n\nRuntime JavaScript package licenses.\n"];
for (const name of [...packages].sort()) {
  const root = path.join("node_modules", name);
  const metadata = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"));
  const licenseFiles = (await readdir(root)).filter(name => /^(license(?:\.(?:md|txt))?|copying)$/i.test(name));
  if (!licenseFiles.length) throw new Error(`Missing license for ${name}`);
  notices.push(`\n${"=".repeat(72)}\n${name} ${metadata.version}\n${"=".repeat(72)}\n`);
  for (const filename of licenseFiles.sort()) notices.push(await readFile(path.join(root, filename), "utf8"));
}
await writeFile("vendor/THIRD_PARTY_LICENSES.txt", notices.join("\n"));
console.log(`Bundled editor and license notices for ${packages.size} packages.`);
