# Source documentation and synchronization

The shared API, CLI, data access, settings, all nine executable stage modules and
the three repository scripts are covered by pdoc. Source functions use Google
style: summary, Args when arguments exist, Returns, Raises for relevant failures,
and an example or command usage. Stage `main()` functions have no parameters;
their file inputs/outputs and prerequisites are specified in module/function docs
and the README. Inline comments explain grouping, alignment, timing boundaries,
clipping and private-estimator inspection.

```powershell
python scripts/build_docs.py
python scripts/build_docs.py --check
python -m http.server 8000 --directory docs/html
```

Open http://localhost:8000/. Pages also work when opened directly from the clone.
GitHub's file viewer shows HTML source; it is not a rendered documentation host.

`docs/source_manifest.json` fingerprints all package modules, support scripts and
generated HTML. A source edit or HTML edit without regeneration fails `--check`.
CRLF/LF is normalized for cross-platform checking. Runtime absolute paths are
hidden from exported constants so page content is portable. CI additionally
regenerates pages and checks Git differences, rather than relying only on hashes.

Before committing a code change, update its docstring, run the relevant tests,
regenerate HTML, run `--check`, and commit the code and documentation together.
Imports are tested to prevent documentation generation from fitting models or
creating experiment output files.
