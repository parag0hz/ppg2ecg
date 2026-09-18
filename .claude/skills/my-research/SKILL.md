---
name: my-research
description: Find and verify facts the way this user does — primary sources first, several search angles, cross-checked numbers, verbatim quotes, and always a 주장·근거·출처 (claim / evidence / source) table. Use this skill whenever the user asks you to look something up, check whether a claim is true, compare tools or methods, find papers or standards, confirm a number, version, price, date or benchmark, or asks "이거 맞아?", "출처 있어?", "문헌 조사", "찾아봐", "알아봐" — even when they don't say the word "research". Also use it before citing any external source in a report, paper draft or advisor update.
---

# my-research

The user's method for finding things out. The point is not speed — it is that **every line can be checked again later
by the user without redoing your search**. A fast answer they cannot verify is worthless to them; a slower answer with
exact quotes and links is reusable.

## The six rules

**1. Primary sources first, blogs last.**
Order: official documentation / standard / spec → the paper itself (arXiv, publisher page, DOI) → the source code or
data itself → survey or review papers → conference talks → blog posts and secondary summaries. A blog is acceptable
for orientation ("what is this thing called?") but never as the source in the table. Secondary sources drift: numbers
get rounded, versions get stale, and the claim mutates as it is copied.

**2. Search from several angles, never one query.**
One query answers the question you already knew how to ask. Run at least three shapes before concluding:
- the user's own words;
- the field's technical term, in English (and the native term if different);
- a site- or domain-restricted search at the primary source (`site:docs...`, `arxiv.org`, the standards body);
- the opposing phrasing — "limitations", "criticism", "failed to reproduce", "retraction" — so you see the
  counter-evidence, not only the confirming page.

**3. Cross-check in at least two independent places — numbers, dates and versions especially.**
Two pages that both copy the same press release are one source, not two. Independent means a different origin: the
paper and the released code; the vendor's docs and a standards document; two research groups. If only one source
exists, that is a finding — mark it `단일 출처` in the table rather than hiding it.

**4. Quote the original sentence; do not paraphrase into the evidence column.**
Paste the sentence as it appears, in its original language, short enough to be an excerpt (roughly one to three
sentences). Fetch the page and read it — a search-result snippet is not the source. This project learned that the
expensive way: a paper's abstract was once reconstructed from search snippets and was wrong, which had to be
disclosed in a report afterwards. A quote you have actually read costs one page fetch; an invented one costs the
user's trust in the whole document.

**5. When you cannot find it, change the angle — never invent.**
Try: a different vocabulary generation (older or newer term), the primary artefact instead of the writing (the
repository, the dataset card, the release notes), a citing paper instead of the original, the author's page, or the
archived version. After a genuine attempt from a few angles, write `확인 못 함` and say what you tried. An honest gap
is usable information; a plausible sentence with no source behind it silently poisons everything downstream.

**6. Always output the same table.**
Same shape every time, so the user can scan it, reuse rows in a paper or a report, and hand it to someone else
without reformatting.

## Output format

Open with one or two sentences of direct answer, then the table:

| 주장 | 근거 (원문 인용) | 출처 |
|---|---|---|
| One checkable statement | "The exact sentence from the source, quoted." | [Title](URL) · 2026-09 · 논문/공식문서/코드 |

Rules for the columns:
- **주장** — one statement per row, specific enough to be wrong. Split compound claims into rows.
- **근거** — the verbatim sentence, in the source's language. Add `(번역: …)` after it only if the user needs it.
  If a claim rests on a number in a table or figure, name the table or figure: `Table III, WildPPG row`.
- **출처** — linked title, the date you read it or the version, and the source type. Mark `단일 출처` when rule 3 could
  not be satisfied, and `2차 출처` when the row rests on a summary rather than the original.

After the table, add only what is needed:
- **확인 못 함** — claims you could not source, and the angles you tried.
- **상충하는 자료** — where sources disagree, with both quotes. Do not silently pick a winner.
- **한 줄 판단** — your reading of the evidence, clearly marked as yours, not the sources'.

## Working notes

Keep the search visible as you go — one short line per angle ("공식 문서에서 버전 확인", "반대 방향으로 재검색") so the
user can see where the coverage is thin.

Cite what you actually used. A row whose source you skimmed but did not open is a rule-4 violation dressed as a
citation.

Scale to the question: a single version number needs one row and two sources, not a literature review. A method
comparison for a paper needs a row per claim, and the counter-evidence angle is mandatory. The table stays the same
either way — that constancy is the point of the format.
