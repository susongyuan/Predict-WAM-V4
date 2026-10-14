#!/usr/bin/env python3
"""Fetch a URL, save html + extracted text under docs/source_cache, append to fetch_index.json."""
import sys, json, re, html, datetime, urllib.request, pathlib
ROOT = pathlib.Path('/home/user/Robotic/project2/docs/source_cache')
IDX = ROOT / 'fetch_index.json'
def extract(h):
    h = re.sub(r'(?is)<(script|style|noscript).*?</\1>', ' ', h)
    h = re.sub(r'(?s)<[^>]+>', ' ', h)
    h = html.unescape(h)
    return re.sub(r'[ \t\r\f\v]+', ' ', re.sub(r'\n\s*\n+', '\n', h)).strip()
def main(sid, url):
    idx = json.load(open(IDX))
    idx = [e for e in idx if e.get('id') != sid]
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research-cache)'})
        with urllib.request.urlopen(req, timeout=40) as r:
            raw = r.read(); status = r.status; resolved = r.geturl()
        h = raw.decode('utf-8', 'replace'); txt = extract(h)
        (ROOT / f'{sid}.html').write_text(h); (ROOT / f'{sid}.txt').write_text(txt)
        e = {'id': sid, 'url': url, 'resolved_url': resolved, 'status': status, 'bytes': len(raw),
             'fetched_utc': now, 'text': txt[:300]}
        print(f'OK {sid} {status} {len(raw)}B :: {txt[:160]}')
    except Exception as ex:
        e = {'id': sid, 'url': url, 'error': str(ex), 'fetched_utc': now}
        print(f'FAIL {sid} {ex}')
    idx.append(e); json.dump(idx, open(IDX, 'w'), indent=2, ensure_ascii=False)
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
