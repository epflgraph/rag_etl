video_understanding_prompt = """

Analyze in-depth the entire given video, and provide:

1. The language used in the video.

2. a general description (in English and in French) of the concepts explained in the video in natural language.
- Phrase this general video description in terms of the content. For example, instead of 'This video lesson provides a detailed analysis of the dynamics of...' write: 'A detailed analysis of the dynamics of...'
- Do not describe what the teacher does, just the concepts presented
- What is the lecture about? Do not start the description by saying 'The video introduces/is about/describes'. Go directly to the point.

3. Two lists (in English and in French) of at least five and at most 10 keywords that describe the contents of the whole video.

4. Create video segments (VideoUnderstandingSegment) ranging from 3 to 180 seconds, depending on the changes in the video. We want to start a new video segment when there is a noticeable change in slides or content.
Rules: 
- Video segments must be consecutive, with no gaps, and the end_time of each segment must match the start_time of the next.
- The end_time of the last video segment should match the video's end.
- Within each video segment, we want to select the key video frame (key_frame_time) with the most visual information.
- This video keyframe will be, in most cases, the last video frame of the video segment (unless you are confident that it should be another frame)

For each video segment (VideoUnderstandingSegment), provide the following:
4.1. The start_time and end_time of the video segment in SRT (hours:minutes:seconds,milliseconds) format
4.2. The time of the selected key_frame of that video segment in SRT (hours:minutes:seconds,milliseconds) format
4.3. booleans about whether the segment contains math and/or diagrams
4.4. boolean indicating whether the teacher uses a pointer to point at different parts of the video while they speak
4.5. Two lists (in English and in French) of at least one and at most 10 keywords that describe the contents of the video segment.

4.6. A short description (in English and French) of the video segment in natural language, following these rules:
- Do not mention what the teacher does, only refer to the content.
- Do not mention on the writing hand or pen of the teacher if these are visible.
- Phrase the description by writing directly about the content. For example, instead of 'This segment details the plan for...' write: 'A detailed plan for...'.
- Ignore if there is a title of the course, name of the university (EPFL), or name of the teacher.
- Ignore if there is a small video window with the teacher.

4.7 The complete audio transcription (in English and in French) from that video segment (from start_time to end_time). For this audio transcription, do not refer to any external sound or music.

4.8. The detailed description of the content of the key text video frame (extracted_text_video_frame), following these rules:

- **Formatting**:
- Write ALWAYS in GitHub-Flavored Markdown.
- Use proper Markdown headings (#, ##, ###) matching the visual hierarchy.
- Always include titles, subtitles, and section headers as part of the Markdown hierarchy.
- Maintain paragraphs, bullet and numbered lists, blockquotes, code blocks, and inline formatting (bold, italics, monospace).
- When representing tables, align columns and retain header rows and data integrity.
- Preserve links and footnotes accurately.

- **Mathematical Content**:
- Preserve mathematical expressions as LaTeX:
    - Inline math: `$ ... $`
    - Block math: `$$ ... $$`
    - When multiple aligned equations are detected, render them as a single block math region within `$$ ... $$` instead.

- **Figures and Images**:
- Describe Figures and Images in detail.
- When there is more than one, mention how many there are and start describing them from left to right and top to bottom.
- Start by describing each image/figure in general. Then continue by describing its parts in detail from left to right and top to bottom.
- When the parts of the image contain labels such as (a), (1) that are described in the video key frame, include their description.
- When the image/figure describes a process or flow, describe how the elements are connected.

- **Code**:
- Use 'triple backticks' Markdown notation, indicating the programming language to separate code from the rest of the content from the video frame. e.g.
```cpp 
# Here is some C++ code in the video frame
using namespace std;
```           

- **General Remarks**:
- When the video frame contains software, do NOT extract text from its interface. e.g., skip words from the navigation menu: File, Preferences, Edit, etc.
- Ignore the presenter's pen or pointer.
- Ignore if there is a title of the course, name of the university (EPFL), or name of the teacher.
- Ignore if there is a small video window with the teacher.
"""
