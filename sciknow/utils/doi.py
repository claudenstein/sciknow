import re

# DOI regex: matches 10.XXXX/anything
_DOI_PATTERN = re.compile(
    r'\b(10\.\d{4,9}/[-._;()/:\w]+)',
    re.IGNORECASE,
)

# arXiv ID patterns: old (hep-ph/9901001 or astro-ph/0207637v1) and new
# (2301.00001 or 2301.00001v2). The OLD pattern must allow a trailing
# version suffix — "astro-ph/0207637v1" is a valid arXiv ID and is
# exactly what shows up in DB rows whose title was set from the
# PDF's first-line arXiv stamp.
_ARXIV_NEW = re.compile(r'\b(\d{4}\.\d{4,5}(?:v\d+)?)\b')
_ARXIV_OLD = re.compile(r'\b([a-z\-]+/\d{7}(?:v\d+)?)\b', re.IGNORECASE)


def extract_doi(text: str) -> str | None:
    match = _DOI_PATTERN.search(text)
    if match:
        doi = match.group(1).rstrip('.')
        return doi
    return None


def extract_arxiv_id(text: str) -> str | None:
    match = _ARXIV_NEW.search(text)
    if match:
        return match.group(1)
    match = _ARXIV_OLD.search(text)
    if match:
        return match.group(1)
    return None


def normalize_doi(doi: str) -> str:
    doi = doi.strip()
    if doi.startswith('https://doi.org/'):
        doi = doi[len('https://doi.org/'):]
    elif doi.startswith('http://doi.org/'):
        doi = doi[len('http://doi.org/'):]
    elif doi.startswith('doi:'):
        doi = doi[4:]
    return doi
