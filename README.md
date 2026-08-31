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

It uses GraphAI to detect the change of slides, drops changes less than ten seconds
apart, and uses ffmpeg to grab one frame per slide. The frame is taken shortly
before the next change, so anything the lecturer wrote on the slide meanwhile
is visible. The video is read straight from its URL and never downloaded.

Each frame becomes its own resource, with a link that opens the recording at
that moment. The video resource is replaced by its frames.

## ImageToMarkdownTransformer

Reads what is on the image (or slide) and writes it out as Markdown.

It sends each image to the RCP vision model, which transcribes the text and figures.

## MergeSlideTranscriptTransformer

Adds to each slide what the lecturer said while it was on screen. 

A slide lasts from its own moment until the next slide starts, so this reads
the subtitles for that time laps and appends them as a quote. 

