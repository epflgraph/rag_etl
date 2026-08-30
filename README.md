# RAG ETL pre-pipeline

# Transformers

## Transformers for Video

The three Transformers below turn a lecture recording into one Markdown chunk per slide,
each linking back to the exact moment of the video it came from. They run in
this order:

```
  video resource
  (Mediaspace subtitles, or a MOOC video descriptor)
        |
        v
  VideoToFramesTransformer
        |   one image per slide, each carrying a link to its moment
        v
  ImageToMarkdownTransformer
        |   the slide read into Markdown
        v
  MergeSlideTranscriptTransformer
        |   plus the words spoken while that slide was up
        v
  one Markdown chunk per slide
```

## VideoToFramesTransformer

Finds the moments where the slide changes and saves a picture of each one.

It asks GraphAI when the slides change, drops changes less than ten seconds
apart, and uses ffmpeg to grab one frame per slide. The frame is taken shortly
before the next change, so anything the lecturer wrote on the slide meanwhile
is visible. The video is read straight from its URL and never downloaded.

Each frame becomes its own resource, with a link that opens the recording at
that moment. The video resource is replaced by its frames.

## ImageToMarkdownTransformer

Reads what is on a slide and writes it out as Markdown.

It sends each image to the vision model, which transcribes the printed text,
the handwritten annotations, tables and formulas, and describes any figure.
Maths is kept as LaTeX. A slide with nothing readable on it is dropped rather
than becoming an empty chunk.

It works on any image resource, not only on frames cut from a video.

## MergeSlideTranscriptTransformer

Adds to each slide what the lecturer said while it was on screen.

A slide lasts from its own moment until the next slide starts, so this reads
the subtitles for that stretch and appends them under a `## Transcript`
heading. The result is one chunk holding both what was written and what was
said, which is what makes a spoken explanation findable.

Slides with no subtitles, or with nothing spoken over them, are left as they
are.
