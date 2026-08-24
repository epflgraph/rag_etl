CLASSIFY_THREAD_SYSTEM_PROMPT = "You are an expert teaching assistant experienced in classifying student questions."

CLASSIFY_THREAD_USER_PROMPT = """
Classify this thread from an educational forum.

Thread metadata:
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

If the type is 'theory', 'practice', or 'exam', also choose a subtype from these options:
{subtype_options}

Extract document numbers if mentioned (e.g., "Homework 7" -> doc_number="7", "Exercise 3.a" -> doc_number="3", doc_subnumber="a").
For exams, doc_number should be the year (e.g., "Exam 2021" -> doc_number="2021").

Provide output as JSON with fields: type, subtype, doc_number, doc_subnumber, week
Set fields to null if not applicable.

{format_instructions}
"""
