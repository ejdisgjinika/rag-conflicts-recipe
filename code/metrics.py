"""Jaccard similarity over complete ingredient phrases, excluding recipe headers."""
import re
import statistics

METRIC_ID = "ingredient_set_jaccard_v1"
METRIC_DEFINITION = {
    "unit": "complete ingredient phrase",
    "separators": "semicolon or newline",
    "normalization": "remove recipe header; casefold; normalize whitespace; strip outer quotes and terminal periods",
    "duplicates": "removed",
    "empty_union_score": 0.0,
    "synonym_or_quantity_normalization": False,
}
_HEADER = re.compile(r"^the\s+ingredients\s+of\s+.+?\s+are\s*:\s*", re.IGNORECASE)


def ingredient_set(sentence: str) -> set[str]:
    """Keep multiword ingredients intact; quantities/descriptors remain significant.

    Commas are part of ingredient phrases, not separators. No stemming,
    stopword removal, synonym matching or quantity conversion is applied.
    """
    body = _HEADER.sub("", sentence.strip().strip('"\'`').strip(), count=1)
    ingredients = set()
    for fragment in re.split(r"[;\n]", body):
        item = " ".join(fragment.casefold().split()).strip('"\'`').rstrip(".").rstrip('"\'`').strip()
        if item:
            ingredients.add(item)
    return ingredients


def sentence_iou(sent1: str, sent2: str) -> float:
    """Return Jaccard similarity between sets of complete ingredient phrases."""
    set1, set2 = ingredient_set(sent1), ingredient_set(sent2)
    union = set1 | set2
    return len(set1 & set2) / len(union) if union else 0.0


def compute_iou_stats(
    response: str, reference_chunks: list[str]
) -> tuple[float, float, float]:
    ious = [sentence_iou(response, chunk) for chunk in reference_chunks] or [0.0]
    mean = statistics.mean(ious)
    stddev = statistics.stdev(ious) if len(ious) > 1 else 0.0
    return max(ious), mean, stddev
