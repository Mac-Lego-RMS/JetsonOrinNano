#!/usr/bin/env python3
"""Build the engineering journal PDF from the Markdown chapters.

    pip install -r docs/requirements.txt
    python docs/build.py            # -> docs/build/journal.pdf
    python docs/build.py --check    # only run the checks, no PDF

The chapters in docs/journal/ are plain Markdown so GitHub renders them as
they are. For the PDF they are joined in file-name order, mermaid blocks are
rendered to images (needs mermaid-cli, see docs/README.md; without it the block
stays as code), Pandoc turns the result into Typst and Typst makes the PDF.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
JOURNAL = DOCS / 'journal'
BUILD = DOCS / 'build'

README_MIN_CHARS = 5000        # rule 7 of the WRO FE general rules

MERMAID_BLOCK = re.compile(r'^```mermaid\s*\n(.*?)^```\s*$', re.S | re.M)
CHAPTER_LINK = re.compile(r'\[([^\]]+)\]\((\d\d-[\w-]+\.md)(#[\w-]+)?\)')
# Rendered as PNG by the browser inside mermaid-cli: typst cannot draw the
# HTML labels of mermaid's SVG, and plain SVG text gets wrapped mid-word.
MERMAID_CONFIG = {'theme': 'neutral'}
MERMAID_SCALE = '3'
# Diagrams are drawn at most 75 % of the text width, and never taller than
# DIAGRAM_MAX_HEIGHT - a tall flowchart at full width runs off the page.
TEXT_WIDTH_CM = 21.0 - 2 * 2.2      # A4 minus the margins in metadata.yaml
DIAGRAM_MAX_HEIGHT_CM = 15.0
# Words that almost never show up in English text. Used to find paragraphs
# that still have to be translated before the submission.
GERMAN_HINT = re.compile(
    r'[äöüß]|\b(und|nicht|wird|werden|wurde|sind|der|die|das|mit|auf|für|'
    r'fuer|bei|eine?|ist|auch|wenn|dass|oder|durch|beim|zum|zur)\b', re.I)


def chapters():
    return sorted(p for p in JOURNAL.glob('*.md'))


def mmdc_command():
    exe = shutil.which('mmdc')
    if exe:
        return [exe]
    local = DOCS / 'node_modules' / '.bin' / 'mmdc'
    if local.exists():
        return [str(local)]
    return None


def render_mermaid(source, mmdc, out_dir):
    """Render one mermaid block to PNG, cached by content hash."""
    digest = hashlib.sha1(source.encode()).hexdigest()[:12]
    img = out_dir / ('mermaid-%s.png' % digest)
    if img.exists():
        return img
    src = out_dir / ('mermaid-%s.mmd' % digest)
    src.write_text(source)
    cfg = out_dir / 'mermaid-config.json'
    cfg.write_text(json.dumps(MERMAID_CONFIG))
    cmd = mmdc + ['-i', str(src), '-o', str(img), '-c', str(cfg), '-b', 'white',
                  '-s', MERMAID_SCALE]
    # Chrome's sandbox needs unprivileged user namespaces, which Ubuntu 24.04
    # (and with it the GitHub runner) blocks - so it is always switched off.
    puppeteer = {'args': ['--no-sandbox']}
    chromium = os.environ.get('PUPPETEER_EXECUTABLE_PATH')
    if chromium:
        puppeteer['executablePath'] = chromium
    pp = out_dir / 'puppeteer.json'
    pp.write_text(json.dumps(puppeteer))
    cmd += ['-p', str(pp)]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        sys.exit('mermaid-cli failed on %s:\n%s' % (src, run.stderr or run.stdout))
    return img


def diagram_width(img):
    """Width in percent of the text width, capped so the height fits."""
    with open(img, 'rb') as fh:
        w, h = struct.unpack('>II', fh.read(24)[16:24])   # PNG IHDR
    fit = 100.0 * DIAGRAM_MAX_HEIGHT_CM * w / h / TEXT_WIDTH_CM
    return int(min(75, fit))


def join_chapters(mmdc):
    mermaid_dir = BUILD / 'mermaid'
    mermaid_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for path in chapters():
        text = path.read_text()

        def replace(match):
            if mmdc is None:
                return match.group(0)
            img = render_mermaid(match.group(1), mmdc, mermaid_dir)
            return '![](%s){width=%d%%}\n' % (os.path.relpath(img, BUILD),
                                              diagram_width(img))

        text = MERMAID_BLOCK.sub(replace, text)
        # Links between chapter files only make sense on GitHub.
        text = CHAPTER_LINK.sub(r'\1', text)
        parts.append(text)
    return '\n\n'.join(parts)


def build_pdf():
    try:
        import pypandoc
        import typst
    except ImportError:
        sys.exit('missing dependencies: pip install -r docs/requirements.txt')

    mmdc = mmdc_command()
    if mmdc is None:
        print('warning: mermaid-cli (mmdc) not found, diagrams stay as code')

    BUILD.mkdir(exist_ok=True)
    markdown = join_chapters(mmdc)
    # Written into build/ so relative image paths (../figures/...) resolve
    # the same way as from journal/.
    typ = BUILD / 'journal.typ'
    pypandoc.convert_text(
        markdown, 'typst', format='markdown', outputfile=str(typ),
        extra_args=['--standalone',
                    '--metadata-file=%s' % (JOURNAL / 'metadata.yaml'),
                    '--resource-path=%s' % JOURNAL,
                    # otherwise pandoc gives every column of a wide table
                    # the same width
                    '--columns=10000'])
    pdf = BUILD / 'journal.pdf'
    typst.compile(str(typ), output=str(pdf), root=str(REPO))
    print('wrote %s' % pdf.relative_to(REPO))


def check():
    """Checks that cost points if they fail. Returns the number of problems."""
    problems = 0
    readme = REPO / 'README.md'
    n = len(readme.read_text()) if readme.exists() else 0
    if n < README_MIN_CHARS:
        print('README.md has %d characters, the rules ask for at least %d'
              % (n, README_MIN_CHARS))
        problems += 1

    for path in [readme] + chapters():
        if not path.exists():
            continue
        in_comment = False
        for no, line in enumerate(path.read_text().splitlines(), 1):
            # Author notes in <!-- --> are not rendered, skip them.
            if '<!--' in line:
                in_comment = True
            if not in_comment and len(GERMAN_HINT.findall(line)) >= 2:
                print('%s:%d: looks German: %s'
                      % (path.relative_to(REPO), no, line.strip()[:70]))
                problems += 1
            if '-->' in line:
                in_comment = False
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true',
                        help='only run the checks')
    args = parser.parse_args()
    problems = check()
    if not args.check:
        build_pdf()
    if problems:
        print('%d problem(s) found' % problems)
    return 1 if args.check and problems else 0


if __name__ == '__main__':
    sys.exit(main())
