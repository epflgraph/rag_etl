
    You will receive multiple Markdown snippets, one per PDF page, enclosed in triple backticks, in strict page order.

    Goal: stitch them into a single clean GitHub-Flavored Markdown document **without mixing distinct sections** (e.g., problem statements vs. solutions), and **without reordering** content.

    Core constraints (must follow all):
    - Preserve page order exactly; do not reorder snippets. Absolutely no cross-page interleaving.
    - Keep all authored content; **do not** summarize, paraphrase, or invent text.
    - Normalize heading levels so hierarchy is consistent across pages (ensure exactly one top-level `#` document title if present).
    - Merge paragraphs/lists if clearly split across two consecutive pages (continue ordered list numbering correctly); fix soft hyphenation at line ends.
    - Merge multi-page tables if a table is clearly split across two consecutive pages; preserve valid GFM table syntax.
    - Merge multi-page display equations if a block is split across two consecutive pages; keep LaTeX integrity.
    - Keep image `![ALT]` items and captions in place; do not generate images.
    - Preserve page boundary markers of the form `<!-- page N -->` exactly as they appear.

    Output **only** the final Markdown (no explanations, metadata, or commentary).
    