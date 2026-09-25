#!/usr/bin/env python3
"""Build data.json, the self-contained index.html, and pareto-front.gif from archived AA observations."""
from __future__ import annotations
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
HISTORY = json.loads((ROOT / 'upstream-history.json').read_text())['models']
BOUNDARY = '2026-09-05'   # first observation on the Intelligence Index version plotted here
END = '2026-09-23'        # last day in the archived snapshot
X_MIN, X_MAX = 0.001, 20.0
Y_MIN, Y_MAX = 0, 65
# Quarter ends for the small multiples; the last panel is the latest archived day.
SNAPSHOTS = ['2024-12-31', '2025-03-31', '2025-06-30', '2025-09-30',
             '2025-12-31', '2026-03-31', '2026-06-30', END]

EFFORT = re.compile(r' \(Adaptive Reasoning, (\w+) Effort(?:, ([^)]+?) Fallback)?\)$')

def nice_name(name):
    """Shorten Artificial Analysis effort suffixes: '(Adaptive Reasoning, Max Effort, Default Fallback)' -> '(max)'."""
    match = EFFORT.search(name)
    if not match:
        return name
    effort, fallback = match.group(1).lower(), match.group(2)
    suffix = effort if fallback in (None, 'Default') else f'{effort}, {fallback} fallback'
    return name[:match.start()] + f' ({suffix})'

# Fix every model at its latest measurement under the same scoring method.
# The boundary excludes retired models without a measurement on that method.
# Dates control when models enter the plot; they never change an existing point.
REFERENCE = {}
for slug, model in HISTORY.items():
    observations = [o for o in model.get('observations', [])
                    if BOUNDARY <= o[0] <= END and o[1] is not None and o[1] > 0 and o[2] is not None]
    if observations and model.get('release_date'):
        cost, score = observations[-1][1:3]
        if math.isfinite(cost) and math.isfinite(score):
            REFERENCE[slug] = (round(cost, 6), round(score, 2))
START = min(HISTORY[slug]['release_date'] for slug in REFERENCE)

models = []
for slug, (cost, score) in REFERENCE.items():
    m = HISTORY[slug]
    models.append({'id': slug, 'name': nice_name(m['name']), 'creator': m.get('creator') or 'Other',
                   'release': m['release_date'], 'cost': cost, 'score': score})
models.sort(key=lambda x: x['id'])
by_id = {m['id']: m for m in models}
assert len({m['name'] for m in models}) == len(models), 'Two plotted models share a display name'
assert all(X_MIN < m['cost'] < X_MAX and Y_MIN <= m['score'] < Y_MAX for m in models), 'A model falls outside the fixed axes'
excluded = {slug for slug in HISTORY if slug not in by_id}
assert len(excluded) == 36 and all(HISTORY[slug].get('retired') for slug in excluded), \
    'Review the page caveat: the excluded-model count or reason changed'

def released(date):
    return [m for m in models if m['release'] <= date]

def frontier(points):
    """Ids of models for which every cheaper plotted model scores lower, cheapest first."""
    ranked = sorted(points, key=lambda p: (p['cost'], -p['score']))
    out, top = [], -float('inf')
    for p in ranked:
        if p['score'] > top + 1e-9:
            out.append(p['id'])
            top = p['score']
    return out

def weekly_dates():
    a, b = dt.date.fromisoformat(START), dt.date.fromisoformat(END)
    out = []
    while a <= b:
        out.append(a.isoformat())
        a += dt.timedelta(days=7)
    return sorted(set(out + [END]))

frames = [{'date': d, 'count': len(released(d)), 'frontier': frontier(released(d))} for d in weekly_dates()]
snapshots = [{'date': d, 'count': len(released(d)), 'frontier': frontier(released(d))} for d in SNAPSHOTS]

payload = {'models': models, 'frames': frames, 'snapshots': snapshots, 'start': START, 'end': END,
           'excluded': len(excluded), 'axes': {'cost': [X_MIN, X_MAX], 'score': [Y_MIN, Y_MAX]}}
(ROOT / 'data.json').write_text(json.dumps(payload, separators=(',', ':')))

# The HTML is one portable file: data, styles and code are embedded.
template = (ROOT / 'template.html').read_text()
assert template.count('/*__DATA__*/') == 1
(ROOT / 'index.html').write_text(template.replace('/*__DATA__*/', json.dumps(payload, separators=(',', ':'))))
if '--no-gif' in sys.argv:  # quick page-only rebuild while editing template.html
    print('wrote data.json and index.html; skipped the GIF')
    sys.exit(0)

# ---------------------------------------------------------------- GIF
# Same data, axes, frontier and palette as the page. System fonts stand in for the web fonts.
W, H = 1200, 760
PLOT = (96, 150, 1150, 648)
S = 3  # draw at 3x, then downsample for smooth lines and text
PAPER = '#fcfbf7'; INK = '#1c1b18'; INK2 = '#57544c'; INK3 = '#8a867b'
RULE = '#e2ded3'; GREY = '#a8a396'; BLUE = '#2f6ea6'

def first_font(*paths):
    return next((str(p) for p in map(Path, paths) if p.exists()), 'DejaVuSans.ttf')
SANS = first_font('/System/Library/Fonts/HelveticaNeue.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
SERIF = first_font('/System/Library/Fonts/Palatino.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf')

def font(size, serif=False):
    return ImageFont.truetype(SERIF if serif else SANS, size * S)

class Canvas:
    def __init__(self):
        self.im = Image.new('RGB', (W * S, H * S), PAPER)
        self.d = ImageDraw.Draw(self.im)
    def text(self, xy, s, size, fill, anchor='ls', serif=False):
        self.d.text((xy[0] * S, xy[1] * S), s, font=font(size, serif), fill=fill, anchor=anchor)
    def box(self, xy, s, size, anchor='ls'):
        b = self.d.textbbox((xy[0] * S, xy[1] * S), s, font=font(size), anchor=anchor)
        return tuple(v / S for v in b)
    def line(self, pts, fill, width=1):
        self.d.line([(x * S, y * S) for x, y in pts], fill=fill, width=round(width * S), joint='curve')
    def dot(self, x, y, r, fill, outline=None):
        self.d.ellipse(((x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S), fill=fill,
                       outline=outline, width=S if outline else 0)
    def image(self):
        return self.im.resize((W, H), Image.Resampling.LANCZOS)

def xy(cost, score):
    l, t, r, b = PLOT
    x = l + (math.log10(cost) - math.log10(X_MIN)) / (math.log10(X_MAX) - math.log10(X_MIN)) * (r - l)
    y = b - (score - Y_MIN) / (Y_MAX - Y_MIN) * (b - t)
    return x, y

def price(v):
    return f'${v:.3g}' if v >= 1 else '$' + f'{v:.3f}'.rstrip('0').rstrip('.')

def step_path(front):
    """The frontier as a staircase: the best score available at or below each cost."""
    pts = []
    for i, p in enumerate(front):
        x, y = xy(p['cost'], p['score'])
        if i:
            pts.append((x, pts[-1][1]))
        pts.append((x, y))
    if pts:
        pts.append((PLOT[2], pts[-1][1]))
    return pts

def base(name):
    return re.sub(r' \([^)]*\)$', '', name)

def draw_frame(fr):
    c = Canvas()
    l, t, r, b = PLOT
    date = dt.date.fromisoformat(fr['date'])
    c.text((56, 62), 'Intelligence versus cost', 34, INK, serif=True)
    c.text((57, 94), 'Artificial Analysis Intelligence Index against cost per Index task, models added by release date', 16, INK2)
    c.text((r, 62), date.strftime('%b %-d, %Y'), 30, INK, anchor='rs', serif=True)
    c.text((r, 94), f"{fr['count']} models released by this date, {len(fr['frontier'])} on the frontier", 16, INK2, anchor='rs')
    for score in (0, 20, 40, 60):
        y = xy(X_MIN, score)[1]
        c.line([(l, y), (r, y)], RULE)
        c.text((l - 10, y + 5), str(score), 14, INK3, anchor='rs')
    for cost in (0.001, 0.01, 0.1, 1, 10):
        x = xy(cost, 0)[0]
        c.line([(x, t), (x, b)], RULE)
        c.text((x, b + 22), price(cost), 14, INK3, anchor='ms')
    c.text((l, t - 14), 'Intelligence Index', 15, INK2)
    c.text((r, b + 50), 'Cost per Index task, US dollars (log scale)', 15, INK2, anchor='rs')
    visible = released(fr['date'])
    front = [by_id[i] for i in fr['frontier']]
    for p in visible:
        c.dot(*xy(p['cost'], p['score']), 3.2, GREY)
    if front:
        c.line(step_path(front), BLUE, 2)
    for p in front:
        c.dot(*xy(p['cost'], p['score']), 4.2, BLUE, outline=PAPER)
    # Label the first model of each run of same-named frontier points, left of the point,
    # where no plotted model can be (it would be cheaper and score higher).
    # Obstacles: frontier dots and the step line. Candidates mirror the page's label placement.
    boxes = [(x - 6, y - 6, x + 6, y + 6) for x, y in (xy(p['cost'], p['score']) for p in front)]
    steps = step_path(front)
    boxes += [(min(a[0], b[0]), min(a[1], b[1]) - 1.5, max(a[0], b[0]), max(a[1], b[1]) + 1.5)
              for a, b in zip(steps, steps[1:])]
    tries = [('rs', -9, 5), ('rs', -8, -9), ('rs', -7, -24), ('rs', -8, 20), ('ls', 9, -9),
             ('rs', -7, -39), ('rs', -7, -54), ('ls', 9, 20)]
    for i, p in enumerate(front):
        if i and base(by_id[front[i - 1]['id']]['name']) == base(p['name']):
            continue
        x, y = xy(p['cost'], p['score'])
        for anchor, dx, dy in tries:
            bb = c.box((x + dx, y + dy), p['name'], 14, anchor=anchor)
            if bb[0] < l + 2 or bb[2] > W - 4 or bb[1] < t - 8 or bb[3] > b:
                continue
            if any(bb[0] < q[2] + 2 and q[0] < bb[2] + 2 and bb[1] < q[3] + 2 and q[1] < bb[3] + 2 for q in boxes):
                continue
            boxes.append(bb)
            c.text((x + dx, y + dy), p['name'], 14, INK, anchor=anchor)
            nx, ny = min(max(x, bb[0]), bb[2]), min(max(y, bb[1] + 3), bb[3] - 3)
            dist = math.hypot(nx - x, ny - y)
            if dist > 14:  # leader line when the label is not right beside its point
                ux, uy = (nx - x) / dist, (ny - y) / dist
                c.line([(x + ux * 6, y + uy * 6), (nx - ux * 3, ny - uy * 3)], INK3, 0.75)
            break
    c.text((56, 736), 'Every dot uses its September 2026 score and cost; dates only decide when a model appears. '
           'Data: Artificial Analysis, via the CatalystNeuro archive.', 13, INK3)
    return c.image()

# The GIF uses the same weekly dates and frontier calculation as the HTML.
images = [draw_frame(fr) for fr in frames]
durations = [170] * len(images)
durations[0] = 900
durations[-1] = 3600  # hold the end state without repeating frames in the file
palette = images[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
frames_p = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
frames_p[0].save(ROOT / 'pareto-front.gif', save_all=True, append_images=frames_p[1:],
                 duration=durations, loop=0, optimize=True, disposal=2)
print(f'{len(models)} plotted models; {len(excluded)} excluded; {len(frames)} frames; {START} to {END}')
print('outputs:', ROOT / 'data.json', ROOT / 'index.html', ROOT / 'pareto-front.gif')
