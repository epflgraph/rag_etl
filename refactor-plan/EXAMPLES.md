1. **Parse a whole PDF into Markdown**
```python
from ragtools.pdf import pdf_to_markdown

md = pdf_to_markdown("polycopie.pdf")
```

**2. Convert a Jupyter notebook into Markdown** *(exists today)*
```python
from ragtools.notebooks import jupyter_to_markdown

jupyter_to_markdown("tp2.ipynb", "tp2.md")
```

**3. Turn a recording into per-slide moments**
```python
from ragtools.video import video_to_slides

slides = video_to_slides(video_url, detector, srt_path="subs.srt")
for s in slides:
    print(s.start, s.end, s.markdown, s.transcript)
```
