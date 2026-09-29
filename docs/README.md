# Documentation

The engineering journal lives here as Markdown, one file per rubric
criterion, so it can be read directly on GitHub. The PDF for the judges
and the printed copy are generated from the same files.

| Path | Content |
|---|---|
| [`journal/`](journal) | chapters, joined in file-name order; `metadata.yaml` holds title and layout |
| `figures/` | figures used by the chapters (plots are generated, photos are copied here) |
| `diagrams/` | larger diagram sources |
| `data/` | measurement data behind the plots (CSV) |
| `analysis/` | scripts that turn the recorded bags into `data/` and `figures/` |
| `build.py` | builds `build/journal.pdf` |

## Writing

- Plain Markdown. Formulas with `$...$` / `$$...$$`, they render on GitHub
  and in the PDF.
- Images relative to the chapter: `![Caption](../figures/name.svg)`. Prefer
  SVG for plots and diagrams, JPG for photos.
- Diagrams as ` ```mermaid ` blocks (flowcharts, state machines, sequence
  diagrams). GitHub renders them; the build turns them into SVG.
- Notes for the authors go into `<!-- ... -->`, they show up nowhere.
- Every claim that has a number needs a source: a figure, a table, a bag or
  a commit.

## Building the PDF

```bash
pip install -r docs/requirements.txt          # pandoc + typst, no LaTeX needed
npm install -g @mermaid-js/mermaid-cli        # optional, for the diagrams
python docs/build.py                          # -> docs/build/journal.pdf
python docs/build.py --check                  # README length, untranslated text
```

On GitHub the PDF is built on every push to `main` (Actions → Engineering
journal → artifact) and attached to the release for every `v*` tag.
