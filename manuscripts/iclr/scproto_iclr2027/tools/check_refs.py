"""Check every reference in a .bib against Crossref (and arXiv as a fallback).

Motivation: the ICLR/NeurIPS Citation Checker flags hallucinated references, and
two hallucinated refs is a presumptive reject. This script does the same job,
repeatably, against authoritative metadata rather than memory.

    python tools/check_refs.py references.bib
    python tools/check_refs.py references.bib --json tools/ref_report.json

What it does per entry:
  * DOI present   -> fetch that exact DOI from Crossref and compare fields.
                     A DOI that resolves to a DIFFERENT paper is the worst
                     failure mode (it is what got scPoli flagged), so it is
                     reported as WRONG-DOI, not as a title mismatch.
  * no DOI        -> search Crossref by title, take the best match.
  * still nothing -> try arXiv (many ML refs are preprints Crossref lacks).

Verdicts:
  OK          every checked field agrees
  CHECK       found the paper, but >=1 field disagrees (listed)
  WRONG-DOI   the DOI resolves, but to a different paper -> fix immediately
  NOT-FOUND   nothing matched anywhere -> verify by hand; possible hallucination
  SKIP        entry has no title to search on

Nothing here edits the .bib; it only reports.
"""
import argparse
import difflib
import io
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

MAILTO = "fatemehs.hashemig@gmail.com"   # Crossref asks for a contact; keeps us in the polite pool
UA = "scproto-refcheck/1.0 (mailto:%s)" % MAILTO
TITLE_MATCH_THRESHOLD = 0.75   # DOI lookup: below this the DOI points elsewhere
# Title-only search is much less trustworthy than a DOI lookup: Crossref happily
# returns a CRAN package named "madgrad" for a query about Adam. So a title-only
# match must clear a higher bar AND share at least one author with the entry.
SEARCH_TITLE_THRESHOLD = 0.90


# ----------------------------------------------------------------- bib parsing
def parse_bib(path):
    """Minimal BibTeX parser: good enough for a hand-maintained references.bib.

    Returns [{'key':..., 'type':..., 'fields':{name: value}}]. Brace-counting
    rather than regex-per-field, so values containing {...} survive intact.
    """
    text = io.open(path, encoding="utf-8").read()
    entries, i = [], 0
    while True:
        at = text.find("@", i)
        if at < 0:
            break
        m = re.match(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text[at:])
        if not m:                       # @comment / stray @
            i = at + 1
            continue
        start_body = at + m.end()
        depth, end = 0, len(text) - 1
        for j in range(at, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        entries.append({"key": m.group(2), "type": m.group(1).lower(),
                        "fields": parse_fields(text[start_body:end])})
        i = end + 1
    return entries


def parse_fields(body):
    fields, i, n = {}, 0, len(body)
    while i < n:
        m = re.compile(r"\s*(\w+)\s*=\s*").match(body, i)
        if not m:
            i += 1
            continue
        name, i = m.group(1).lower(), m.end()
        if i < n and body[i] in "{\"":
            opener = body[i]
            depth, start = 0, i + 1
            j = i
            for j in range(i, n):
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                    if depth == 0 and opener == "{":
                        break
                elif body[j] == "\"" and opener == "\"" and j > start:
                    break
            fields[name] = body[start:j]
            i = j + 1
        else:                                   # bare value: year = 2023
            m2 = re.compile(r"([^,]*)").match(body, i)
            fields[name] = m2.group(1).strip()
            i = m2.end()
        comma = body.find(",", i)
        i = n if comma < 0 else comma + 1
    return fields


# ------------------------------------------------------------------- normalise
# LaTeX commands that ARE a letter rather than an accent on one. Stripping
# these as commands turns St{\aa}hl into "st hl" and flags Stahl as an impostor.
LATEX_LETTERS = [(r"\aa", "a"), (r"\AA", "A"), (r"\ae", "ae"), (r"\AE", "AE"),
                 (r"\oe", "oe"), (r"\OE", "OE"), (r"\ss", "ss"),
                 (r"\o", "o"), (r"\O", "O"), (r"\l", "l"), (r"\L", "L"),
                 (r"\i", "i"), (r"\j", "j")]


# Latin letters with a stroke: NFKD does NOT decompose these, so they survive
# as non-ASCII and then get deleted, turning Szalata into "szaata".
STROKE_LETTERS = str.maketrans({
    "ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D",
    "ħ": "h", "ŧ": "t", "ı": "i", "ß": "ss", "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE", "ð": "d", "þ": "th",
})


def clean(s):
    r"""Fold LaTeX and Unicode so all of these compare equal:
    Salm{\'e}n / Salmen / Salmén,  St{\aa}hl / Stahl,  Pe'er / Pe’er,
    Sza{\l}ata / Szałata,  Egea~Lavandera / Egea Lavandera.
    """
    s = s.replace("’", "'").replace("‘", "'")   # curly -> ASCII quote
    s = s.translate(STROKE_LETTERS)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))   # Salmén -> Salmen
    for cmd, letter in LATEX_LETTERS:                # \aa -> a, before the strip
        s = re.sub(re.escape(cmd) + r"(?![a-zA-Z])", letter, s)
    # Accent commands whose name is punctuation: \'o \`e \^a \"u \~n \=a \.z.
    # Must go before the \~ -> space rule below, or Peri{\~n}an becomes "peri nan".
    s = re.sub(r"""\\['`^"~=.]""", "", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)               # remaining control sequences
    s = s.replace("~", " ")                          # BibTeX hard space in names
    s = re.sub(r"[{}\\$\"'`^]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return " ".join(s.split())


def sim(a, b):
    return difflib.SequenceMatcher(None, clean(a), clean(b)).ratio()


def bib_authors(field):
    """BibTeX 'A and B and others' -> family names, lowercased."""
    out = []
    for part in re.split(r"\s+and\s+", field or ""):
        part = part.strip()
        if not part or clean(part) == "others":
            continue
        fam = part.split(",")[0].strip() if "," in part else part.split()[-1]
        out.append(clean(fam))
    return [a for a in out if a]


# ----------------------------------------------------------------------- fetch
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def crossref_by_doi(doi):
    try:
        return json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    except Exception:
        return None


def datacite_by_doi(doi):
    """Datasets (figshare, Zenodo) register with DataCite, not Crossref.

    Returned in Crossref's shape so the rest of the checker does not care.
    """
    try:
        a = json.loads(get("https://api.datacite.org/dois/" + urllib.parse.quote(doi)))["data"]["attributes"]
    except Exception:
        return None
    year = a.get("publicationYear") or 0
    return {"title": [a["titles"][0]["title"]] if a.get("titles") else [""],
            "container-title": [a.get("publisher") or ""],
            "issued": {"date-parts": [[year]]},
            "author": [{"family": (c.get("familyName") or c.get("name", "").split(",")[0])}
                       for c in a.get("creators", [])],
            "DOI": a.get("doi"), "volume": None, "page": None}


def crossref_by_title(title, want_authors):
    """Best Crossref hit for a title, but only if it is convincingly the same paper.

    want_authors: family names from the bib entry. A candidate is rejected when
    both sides list authors and none overlap -- that is how "Adam: A Method for
    Stochastic Optimization" stops matching the unrelated madgrad CRAN package.
    """
    try:
        url = ("https://api.crossref.org/works?rows=5&query.bibliographic="
               + urllib.parse.quote(title))
        items = json.loads(get(url))["message"]["items"]
    except Exception:
        return None
    best, best_score = None, 0.0
    for it in items:
        t = (it.get("title") or [""])[0]
        s = sim(title, t)
        if s < SEARCH_TITLE_THRESHOLD or s <= best_score:
            continue
        fams = [clean(a.get("family", "")) for a in it.get("author", []) if a.get("family")]
        if want_authors and fams and not set(want_authors) & set(fams):
            continue                      # same-ish words, different people
        best, best_score = it, s
    return best


def arxiv_by_title(title):
    try:
        url = ("http://export.arxiv.org/api/query?max_results=5&search_query=ti:"
               + urllib.parse.quote('"%s"' % title))
        root = ET.fromstring(get(url))
    except Exception:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in root.findall("a:entry", ns):
        t = " ".join((e.findtext("a:title", "", ns) or "").split())
        if sim(title, t) >= SEARCH_TITLE_THRESHOLD:
            year = (e.findtext("a:published", "", ns) or "")[:4]
            return {"title": [t],
                    "container-title": ["arXiv"],
                    "issued": {"date-parts": [[int(year) if year.isdigit() else 0]]},
                    "author": [{"family": (a.findtext("a:name", "", ns) or "x").split()[-1]}
                               for a in e.findall("a:author", ns)],
                    "DOI": None, "volume": None, "page": None}
    return None


def xr_field(rec, name):
    v = rec.get(name)
    if isinstance(v, list):
        return v[0] if v else None
    return v


# ---------------------------------------------------------------------- verify
def load_manual(path):
    """Keys a human verified by hand, with the evidence URL.

    Some real references are in no machine-readable index: NeurIPS/ICML
    proceedings papers carry no DOI, and workshop papers live only on
    OpenReview. Those come back NOT-FOUND forever, which trains you to ignore
    the category -- exactly the wrong habit. Recording the human check here
    keeps NOT-FOUND meaning "nobody has verified this yet".

    Format, one per line:  key <whitespace> evidence-URL   ('#' comments ok)
    """
    out = {}
    try:
        for line in io.open(path, encoding="utf-8"):
            line = line.split("#")[0].strip()
            if line:
                parts = line.split(None, 1)
                out[parts[0]] = parts[1].strip() if len(parts) > 1 else "(no URL given)"
    except IOError:
        pass
    return out


def check(entry, manual=None):
    f = entry["fields"]
    title = f.get("title", "").strip()
    res = {"key": entry["key"], "title": title, "verdict": "SKIP",
           "problems": [], "suggestions": [], "source": None}
    if not title:
        res["problems"].append("entry has no title field")
        return res

    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", (f.get("doi") or "").strip())
    mine = bib_authors(f.get("author", ""))
    venue_text = clean((f.get("journal") or "") + " " + (f.get("booktitle") or "")
                       + " " + (f.get("eprint") or ""))
    is_preprint = "arxiv" in venue_text or "biorxiv" in venue_text
    rec, source = None, None

    if doi:
        rec = crossref_by_doi(doi)
        source = "Crossref/DOI"
        if rec is None:
            rec = datacite_by_doi(doi)          # figshare / Zenodo datasets
            source = "DataCite/DOI"
        if rec is not None and sim(title, xr_field(rec, "title") or "") < TITLE_MATCH_THRESHOLD:
            res["verdict"] = "WRONG-DOI"
            res["source"] = source
            res["problems"].append("DOI %s resolves to a DIFFERENT paper: %r"
                                   % (doi, xr_field(rec, "title")))
            return res
        if rec is None:
            res["problems"].append("DOI %s does not resolve in Crossref" % doi)

    # For entries the bib itself calls a preprint, ask arXiv before Crossref:
    # Crossref's index is full of same-titled CRAN packages and review records.
    order = ["arxiv", "crossref"] if is_preprint else ["crossref", "arxiv"]
    for step in order:
        if rec is not None:
            break
        if step == "crossref":
            rec, source = crossref_by_title(title, mine), "Crossref/title"
        else:
            rec, source = arxiv_by_title(title), "arXiv"
    if rec is None:
        if manual and entry["key"] in manual:
            res["verdict"] = "MANUAL-OK"
            res["source"] = "hand-checked"
            res["suggestions"].append("no API record; verified by hand: %s" % manual[entry["key"]])
        else:
            res["verdict"] = "NOT-FOUND"
            res["problems"].append("no Crossref, DataCite or arXiv record matches this title")
        return res

    res["source"] = source
    res["found_title"] = xr_field(rec, "title")
    res["found_doi"] = rec.get("DOI")

    # --- year
    try:
        real_year = rec["issued"]["date-parts"][0][0]
    except Exception:
        real_year = None
    bib_year = re.sub(r"\D", "", f.get("year", ""))
    if not bib_year:
        res["problems"].append("year field is EMPTY; real year is %s" % real_year)
    elif real_year and str(real_year) != bib_year:
        # Journals routinely publish online one year before the print issue, and
        # ML papers hit arXiv the year before the conference. A one-year gap is
        # normal and is not what the checker means by a wrong year.
        if abs(int(bib_year) - real_year) <= 1:
            res["suggestions"].append("year: bib=%s  record=%s (online vs issue?)"
                                      % (bib_year, real_year))
        else:
            res["problems"].append("year: bib=%s  real=%s" % (bib_year, real_year))

    # --- venue. Only meaningful when we matched a real published record;
    # an arXiv match says nothing about where it was later published.
    real_venue = xr_field(rec, "container-title")
    bib_venue = f.get("journal") or f.get("booktitle") or ""
    if source != "arXiv" and real_venue and bib_venue and sim(bib_venue, real_venue) < 0.6:
        # Springer registers ECCV/MICCAI papers under the book SERIES name, so
        # "ECCV" vs "Lecture Notes in Computer Science" is correct on both
        # sides. Citing the conference is standard practice, not an error.
        if "lecture notes" in clean(real_venue):
            res["suggestions"].append("venue: bib=%r, registered under the series %r"
                                      % (bib_venue, real_venue))
        else:
            res["problems"].append("venue: bib=%r  real=%r" % (bib_venue, real_venue))

    # --- authors (family names only, order-insensitive)
    real_fams = [clean(a.get("family", "")) for a in rec.get("author", []) if a.get("family")]
    truncated = "others" in clean(f.get("author", "")).split()
    # "Egea Lavandera" vs "Lavandera", "van den Oord" vs "Oord": treat two
    # surnames as the same person when either is contained in the other.
    def known(a):
        return any(a == r or a in r or r in a for r in real_fams)

    if real_fams and mine:
        extra = [a for a in mine if not known(a)]
        if extra:
            res["problems"].append("author(s) NOT on the real paper: %s" % ", ".join(extra))
        missing = [r for r in real_fams if not any(r == a or r in a or a in r for a in mine)]
        # Missing authors only matter when the entry does not say "and others".
        if missing and not truncated and len(missing) > max(1, 0.25 * len(real_fams)):
            res["problems"].append("missing %d/%d authors: %s"
                                   % (len(missing), len(real_fams), ", ".join(missing)))

    # --- DOI present on the real record but absent from the bib. Not an error
    # (the checker does not require DOIs), just a cheap way to make an entry
    # verifiable, so it is a suggestion and does not downgrade the verdict.
    if not doi and rec.get("DOI"):
        res["suggestions"].append("no doi field; real DOI is %s" % rec["DOI"])

    # --- volume / pages, only when the bib bothers to state them.
    # Pages are a weak signal: Crossref often keeps the online-first pagination
    # ("1-27") forever while the printed issue says 2881-2907, so a mismatch
    # here is a prompt to look, never a verdict.
    for bibname, xrname, hard in (("volume", "volume", True), ("pages", "page", False)):
        bv, rv = f.get(bibname), rec.get(xrname)
        if bv and rv:
            b, r = clean(bv).replace(" ", ""), clean(rv).replace(" ", "")
            if b not in r and r not in b:
                msg = "%s: bib=%r  record=%r" % (bibname, bv, rv)
                (res["problems"] if hard else res["suggestions"]).append(msg)

    res["verdict"] = "OK" if not res["problems"] else "CHECK"
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bib", nargs="?", default="references.bib")
    ap.add_argument("--json", help="also write the full report here")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between API calls")
    ap.add_argument("--manual", default="tools/verified_manually.txt",
                    help="keys verified by hand (key <space> evidence-URL per line)")
    a = ap.parse_args()

    manual = load_manual(a.manual)
    entries = parse_bib(a.bib)
    print("Checking %d entries in %s  (%d hand-verified keys loaded)\n"
          % (len(entries), a.bib, len(manual)))
    results = []
    marks = {"OK": "ok", "CHECK": "!!", "WRONG-DOI": "XX", "NOT-FOUND": "??",
             "SKIP": "--", "MANUAL-OK": "mm"}
    for i, e in enumerate(entries, 1):
        r = check(e, manual)
        results.append(r)
        print("[%2d/%d] %s %-10s %-26s  [%s]" % (i, len(entries), marks[r["verdict"]],
                                                 r["verdict"], r["key"], r["source"] or "-"))
        for p in r["problems"]:
            print("           ! %s" % p)
        for s in r.get("suggestions", []):
            print("           . %s" % s)
        time.sleep(a.delay)

    print("\n" + "=" * 72)
    for v in ("WRONG-DOI", "NOT-FOUND", "CHECK", "SKIP", "MANUAL-OK", "OK"):
        keys = [r["key"] for r in results if r["verdict"] == v]
        if keys:
            print("%-10s %2d  %s" % (v, len(keys), ", ".join(keys)))
    print("=" * 72)
    print("WRONG-DOI and NOT-FOUND are what the conference checker calls")
    print("hallucinated. CHECK entries are real papers with wrong metadata --")
    print("not hallucination, but reviewers are told to flag them too.")

    if a.json:
        io.open(a.json, "w", encoding="utf-8").write(json.dumps(results, indent=2))
        print("\nfull report -> %s" % a.json)

    return 1 if any(r["verdict"] in ("WRONG-DOI", "NOT-FOUND") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
