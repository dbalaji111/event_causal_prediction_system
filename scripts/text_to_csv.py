import re
import os
import csv

# --------------------------------------------------------------------------------
# CONFIGURE THESE PATHS
IN_DIR  = r"C:\Users\balaj\code_files\Documents\Brahmanda\context_aware_risk_methodology\event_causal_prediction_system\scripts\pdf_to_text data"
OUT_DIR = r"C:\Users\balaj\code_files\Documents\Brahmanda\context_aware_risk_methodology\event_causal_prediction_system\data"
# --------------------------------------------------------------------------------

HEADLINE_RE = re.compile(r'^(.+?)\.{10,}\s*(\d+)\s*$', re.UNICODE)
DATE_RE     = re.compile(
    r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|'
    r'July|August|September|October|November|December)\s+\d{4})\b'
)

def split_pages(text):
    parts = re.split(r'^--- Page \d+ ---\s*', text, flags=re.M)
    if parts and not parts[0].strip():
        parts = parts[1:]
    return [''] + parts  # pages[1] == first page

def extract_toc(pages):
    toc = []
    for pnum in range(1, len(pages)):
        hits = [HEADLINE_RE.match(l) for l in pages[pnum].splitlines()]
        hits = [m for m in hits if m]
        if not hits:
            break
        for m in hits:
            toc.append({'title': m.group(1).strip(), 'page': int(m.group(2))})
    return toc

def extract_date(text):
    m = DATE_RE.search(text)
    return m.group(1) if m else ''

def process_file(txt_path):
    text  = open(txt_path, encoding='utf-8').read()
    pages = split_pages(text)
    toc   = extract_toc(pages)

    clean, errors = [], []
    for idx, entry in enumerate(toc):
        start = entry['page']
        end   = toc[idx+1]['page'] if idx+1 < len(toc) else len(pages)
        content = "\n".join(pages[start:end]).strip()

        rec = {
            'title':   entry['title'],
            'date':    extract_date(content),
            'source':  "The Wall Street Journal",
            'content': content
        }

        # validate
        if rec['date'] and len(rec['content']) > 200:
            clean.append(rec)
        else:
            errors.append(rec)

    return clean, errors

def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['title','date','source','content'])
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    all_clean, all_errors = [], []

    for fn in os.listdir(IN_DIR):
        if not fn.lower().endswith('.txt'):
            continue
        in_path = os.path.join(IN_DIR, fn)
        clean, errs = process_file(in_path)
        all_clean.extend(clean)
        all_errors.extend(errs)
        print(f"Processed {fn}:  ✓ {len(clean)} clean, ⚠ {len(errs)} errors")

    # write one big master CSV
    write_csv(all_clean, os.path.join(OUT_DIR, "all_clean_articles.csv"))

    # (optional) write one combined errors CSV
    if all_errors:
        write_csv(all_errors, os.path.join(OUT_DIR, "all_error_articles.csv"))

    print(f"\n⇒ Written {len(all_clean)} total articles to all_clean_articles.csv")
    if all_errors:
        print(f"⇒ Written {len(all_errors)} total error rows to all_error_articles.csv")
