from pydantic import BaseModel


class VideoUnderstandingSegment(BaseModel):
    start_time: str
    end_time: str
    key_frame_time: str
    contains_math: bool
    contains_diagram: bool
    teacher_uses_pointer: bool
    segment_audio_transcription_en: str
    segment_audio_transcription_fr: str
    extracted_text_video_frame: str
    short_description_video_segment_en: str
    short_description_video_segment_fr: str
    segment_keywords_en: list[str]
    segment_keywords_fr: list[str]


class VideoUnderstandingResponse(BaseModel):
    language: str
    general_description_en: str
    general_description_fr: str
    video_keywords_en: list[str]
    video_keywords_fr: list[str]
    video_segments: list[VideoUnderstandingSegment]
