CLASSIFY_THREAD_SYSTEM_PROMPT = "You are an expert teaching assistant experienced in classifying student questions."

CLASSIFY_THREAD_USER_PROMPT = """
Classify this thread from an educational forum.

Thread metadata:
- Title: '{thread_title}'
- Category: '{thread_category}'
- Subcategory: '{thread_subcategory}'

Thread content:
{all_messages_html}

Choose exactly one type from: {all_types}

Type descriptions:
- **theory**: Questions about course theory, concepts, definitions, lecture content, slides, or notes. NOT about exercises or assignments.
- **practice**: Questions about homework, exercises, series, labs, projects, assignments, or problem sets.
- **exam**: Questions about previous year exams or exam solutions. NOT about upcoming exams or exam policies.
- **logistics**: Questions about schedules, deadlines, grades, course organization, or policies.
- **bug_or_typo_report**: Reports of errors, typos, or bugs in course materials.
- **exception_request**: Requests for deadline extensions or special accommodations.
- **admin**: Course announcements or administrative messages.
- **other**: Anything not covered above (gratitude, off-topic, general remarks).

If the type is 'theory', 'practice', or 'exam', the thread is about one of the following course
documents, each shown with its catalogue id in brackets:

{catalogue}

If the type is theory, practice or exam, set catalogue_id to the id of the entry the thread is
about. Pick the most specific entry (a specific exercise) when one is discussed; pick the
document-level entry (no sub-number) when the thread is about the document as a whole; set
catalogue_id to null when no entry applies. For any other type, catalogue_id must be null.

The number of an exam or midterm entry is the year assigned by the course staff and may differ
from the year students use to refer to it. Match a thread's exam reference to the entry whose
title (usually the file name, e.g. "Examen_2024_2025") covers that exam, not to the entry whose
number merely equals the year mentioned.

The thread was written during academic year {academic_year}. When it refers to a midterm or exam
without naming a year (e.g. "test mi-semestre ex 1a"), it means the one of that academic year:
prefer the entry numbered {exam_year}.

Set mentioned_number to the document number the thread explicitly refers to (the exam year, e.g.
"2020" for "examen 2020"; the series or homework number, e.g. "3" for "série 3"), even when no
catalogue entry exists for it. Set it to null when the thread does not name any specific document.

Also determine the course week the thread relates to, if it can be determined from the content;
otherwise set week to null.

In `reason`, give the evidence from the thread text that supports your chosen type and catalogue
entry, in a sentence or two. In `confidence`, rate from 0 to 10 how sure you are that both the
type and the catalogue entry you picked are correct.
"""
