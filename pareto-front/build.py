#!/usr/bin/env python3
"""Build a self-contained HTML animation and an animated GIF from archived AA observations."""
from __future__ import annotations
import datetime as dt
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
HISTORY = json.loads((ROOT / 'upstream-history.json').read_text())['models']
BOUNDARY = '2026-09-05'
END = '2026-09-23'
W, H = 1200, 760
PLOT = (108, 154, 878, 660)
X_MIN, X_MAX = 0.003, 20.0
Y_MIN, Y_MAX = 0, 70
RENDER_SCALE = 3
BG = '#fbfbf8'; FG = '#1d2b2d'; MUTED = '#58696b'; GRID = '#dce4e1'
FRONT = '#117965'; POINT = '#aebfbd'; GOLD = '#9c652e'
FONT_PATH=next((str(p) for p in (Path('/System/Library/Fonts/HelveticaNeue.ttc'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')) if p.exists()), 'DejaVuSans.ttf')
SERIF_PATH=next((str(p) for p in (Path('/System/Library/Fonts/NewYork.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf')) if p.exists()), 'DejaVuSerif.ttf')

def font(size, bold=False, serif=False):
    if serif:
        return ImageFont.truetype(SERIF_PATH, size * RENDER_SCALE)
    return ImageFont.truetype(FONT_PATH, size * RENDER_SCALE, index=1 if bold else 0)

class ScaledDraw:
    """Draw at 3x resolution, then downsample for smooth GIF lines and text."""
    def __init__(self, image):
        self.draw = ImageDraw.Draw(image)

    @staticmethod
    def coords(values):
        return tuple(round(value * RENDER_SCALE) for value in values)

    def text(self, position, value, **kwargs):
        self.draw.text(self.coords(position), value, **kwargs)

    def textbbox(self, position, value, **kwargs):
        box = self.draw.textbbox(self.coords(position), value, **kwargs)
        return tuple(value / RENDER_SCALE for value in box)

    def line(self, points, **kwargs):
        kwargs['width'] = max(1, kwargs.get('width', 1) * RENDER_SCALE)
        if points and isinstance(points[0], (tuple, list)):
            points = [self.coords(point) for point in points]
        else:
            points = self.coords(points)
        self.draw.line(points, **kwargs)

    def ellipse(self, box, **kwargs):
        kwargs['width'] = max(1, kwargs.get('width', 1) * RENDER_SCALE)
        self.draw.ellipse(self.coords(box), **kwargs)

    def rounded_rectangle(self, box, **kwargs):
        kwargs['radius'] = kwargs.get('radius', 0) * RENDER_SCALE
        self.draw.rounded_rectangle(self.coords(box), **kwargs)

def nice_name(name):
    name=name.replace(' (Adaptive Reasoning, Max Effort, Default Fallback)', ' (max)')
    name=name.replace(' (Adaptive Reasoning, Xhigh Effort, Default Fallback)', ' (xhigh)')
    name=name.replace(' (Adaptive Reasoning, High Effort, Default Fallback)', ' (high)')
    name=name.replace(' (Adaptive Reasoning, Medium Effort, Default Fallback)', ' (medium)')
    name=name.replace(' (Adaptive Reasoning, Max Effort)', ' (max)')
    name=name.replace(' (Adaptive Reasoning, Xhigh Effort)', ' (xhigh)')
    name=name.replace(' (Adaptive Reasoning, High Effort)', ' (high)')
    return name

# Fix every model at its latest measurement under the same scoring method.
# Dates control when models enter the plot; they never change an existing point.
REFERENCE = {}
for slug, model in HISTORY.items():
    observations = [o for o in model.get('observations', [])
                    if BOUNDARY <= o[0] <= END and o[1] > 0 and o[2] is not None]
    if observations and model.get('release_date'):
        cost, score = observations[-1][1:3]
        if math.isfinite(cost) and math.isfinite(score):
            REFERENCE[slug] = (round(cost, 6), round(score, 2))
START = min(HISTORY[slug]['release_date'] for slug in REFERENCE)

def frontier(points):
    ranked=sorted(points,key=lambda p:(p['cost'],-p['score']))
    out=[];top=-float('inf')
    for p in ranked:
        if p['score']>top+1e-9:
            out.append(p['id']);top=p['score']
    return out

def dates():
    a=dt.date.fromisoformat(START); b=dt.date.fromisoformat(END)
    out=[]
    while a<=b:
        out.append(a.isoformat());a+=dt.timedelta(days=7)
    out.append(END)
    return sorted(set(out))

models=[]
for slug in REFERENCE:
    m=HISTORY[slug]
    models.append({'id':slug,'name':nice_name(m['name']),'creator':m.get('creator') or 'Other','release':m.get('release_date'),'last':m.get('last_seen')})
models.sort(key=lambda x:x['id'])
by_id={m['id']:m for m in models}
frames=[]
for date in dates():
    pts=[]
    for slug,(cost,score) in REFERENCE.items():
        if HISTORY[slug]['release_date'] <= date:
            pts.append({'id':slug,'cost':cost,'score':score})
    ids=frontier(pts)
    frames.append({'date':date,'points':pts,'frontier':ids})

payload={'models':models,'frames':frames,'start':START,'end':END,
         'axes':{'cost':[X_MIN,X_MAX],'score':[Y_MIN,Y_MAX]}}
(ROOT/'data.json').write_text(json.dumps(payload,separators=(',',':')))

# HTML is one portable file: all observations, styling and animation logic are embedded.
template=(ROOT/'template.html').read_text()
(ROOT/'index.html').write_text(template.replace('/*__DATA__*/',json.dumps(payload,separators=(',',':'))))

def xy(cost,score):
    l,t,r,b=PLOT
    x=l+(math.log10(cost)-math.log10(X_MIN))/(math.log10(X_MAX)-math.log10(X_MIN))*(r-l)
    y=b-(score-Y_MIN)/(Y_MAX-Y_MIN)*(b-t)
    return int(x),int(y)

def price(v):
    return f'${v:.3g}' if v>=1 else f'${v:.3f}'.rstrip('0').rstrip('.')

def draw_frame(fr):
    im=Image.new('RGB',(W*RENDER_SCALE,H*RENDER_SCALE),BG);d=ScaledDraw(im)
    d.text((58,31),'Intelligence versus cost',font=font(36,serif=True),fill=FG)
    d.text((59,84),'Artificial Analysis Intelligence Index · Models with reported task cost',font=font(17),fill=MUTED)
    l,t,r,b=PLOT
    for score in range(0,71,10):
        _,y=xy(X_MIN,score)
        d.line((l,y,r,y),fill=GRID,width=1)
        d.text((70,y-9),str(score),font=font(15),fill=MUTED)
    d.line((l,t,l,b),fill=GRID,width=1)
    for cost in [.003,.01,.03,.1,.3,1,3,10,20]:
        x,_=xy(cost,0)
        d.line((x,b,x,b+5),fill=MUTED,width=1)
        label=price(cost);bb=d.textbbox((0,0),label,font=font(14))
        d.text((x-(bb[2]-bb[0])/2,b+14),label,font=font(14),fill=MUTED)
    d.text((l,t-29),'Intelligence Index',font=font(16),fill=MUTED)
    d.text((l+230,b+45),'Cost per Index task (USD, log scale)',font=font(16),fill=MUTED)
    m={p['id']:p for p in fr['points']};front=[m[i] for i in fr['frontier'] if i in m]
    for p in fr['points']:
        x,y=xy(p['cost'],p['score'])
        if l<=x<=r and t<=y<=b:d.ellipse((x-2.8,y-2.8,x+2.8,y+2.8),fill=POINT)
    if len(front)>1:
        d.line([xy(p['cost'],p['score']) for p in front],fill=FRONT,width=3,joint='curve')
    for p in front:
        x,y=xy(p['cost'],p['score'])
        d.ellipse((x-4.5,y-4.5,x+4.5,y+4.5),fill=FRONT,outline=BG,width=1)
    # Keep the direct labels above and left of their frontier points.
    chosen=[]
    if front:
        chosen.append(front[-1])
        if len(front)>1:chosen.append(front[0])
        if len(front)>6:chosen.append(front[len(front)//2])
    occupied=[]
    for p in chosen:
        x,y=xy(p['cost'],p['score']);name=by_id[p['id']]['name']
        name=name[:23]+'…' if len(name)>24 else name
        label_width=d.textbbox((0,0),name,font=font(14))[2]
        tx=max(8,x-label_width-12);ty=y-31
        for _ in range(5):
            if all(abs(tx-a)>110 or abs(ty-c)>22 for a,c in occupied):break
            ty-=22
        occupied.append((tx,ty))
        d.text((tx,ty),name,font=font(14),fill=FG)
    sx=922
    d.text((sx,155),dt.date.fromisoformat(fr['date']).strftime('%b %-d, %Y'),font=font(29,serif=True),fill=FG)
    d.text((sx,201),'Models appear at release',font=font(15),fill=FRONT)
    d.text((sx,259),f"{len(fr['points'])} {'model' if len(fr['points'])==1 else 'models'} shown",font=font(19),fill=FG)
    d.text((sx,289),f"{len(front)} on plotted frontier",font=font(19),fill=FG)
    if front:
        lead=max(front,key=lambda p:p['score'])
        d.text((sx,362),'Highest plotted score',font=font(15),fill=MUTED)
        d.text((sx,391),f"{lead['score']:.1f}  ·  {price(lead['cost'])} per task",font=font(21),fill=FG)
        name=by_id[lead['id']]['name'];name=name[:29]+'…' if len(name)>30 else name
        d.text((sx,423),name,font=font(14),fill=MUTED)
    note=['Dots use September 2026 data.', 'Many higher-scoring 2025 models', 'have no reported Index task cost', 'and cannot be plotted here.']
    for j,line in enumerate(note):d.text((sx,510+23*j),line,font=font(14),fill=MUTED)
    d.text((sx,636),'Existing dots stay in place',font=font(13),fill=FRONT)
    d.text((58,735),'Data: Artificial Analysis via CatalystNeuro · Frontier: higher score and lower cost',font=font(13),fill=MUTED)
    return im.resize((W,H),Image.Resampling.LANCZOS)

# The GIF uses the same discrete dates and frontier calculation as the HTML.
images=[draw_frame(fr) for fr in frames]
# Hold the end state for two seconds without repeating full frames in the file.
durations=[160 + 10 * (i % 2) for i in range(len(images))]
durations[0]=900;durations[-1]=3300
palette=images[-1].quantize(colors=256,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE)
palette_images=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in images]
palette_images[0].save(ROOT/'pareto-front.gif',save_all=True,append_images=palette_images[1:],duration=durations,loop=0,optimize=True,disposal=2)
print(f'{len(models)} archived models; {len(frames)} frames; {START} to {END}')
print('outputs:',ROOT/'index.html',ROOT/'pareto-front.gif')
