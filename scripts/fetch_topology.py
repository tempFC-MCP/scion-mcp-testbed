"""Collect a fresh RIPE Atlas and RIPEstat sample for the public-routing experiment."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from tempfile import mkdtemp
from urllib.request import urlopen

DATA = Path(__file__).resolve().parents[1] / 'data' / 'ripe'
ATLAS = 'https://atlas.ripe.net/api/v2/probes/?status=1&is_public=true&page_size=500'
RIS = 'https://stat.ripe.net/data/bgp-state/data.json?resource='
SEED = 6705


def download(url, target):
    with urlopen(url, timeout=45) as response:
        raw = response.read()
    document = json.loads(raw)
    target.write_bytes(raw)
    return document, dict(url=url, file=target.name, sha256=hashlib.sha256(raw).hexdigest())


def collect():
    DATA.mkdir(parents=True, exist_ok=True)
    raw = Path(mkdtemp(prefix='download-', dir=DATA))
    atlas, atlas_source = download(ATLAS, raw / 'atlas-first-page.json')
    probes = [probe for probe in atlas['results']
              if probe.get('prefix_v4') and probe.get('asn_v4')
              and (probe.get('geometry') or {}).get('coordinates')]
    if len(probes) < 50:
        raise ValueError(f'Needed 50 eligible Atlas probes in the first page, found {len(probes)}')
    random.Random(SEED).shuffle(probes)
    selected = probes[:50]

    def fetch_routes(prefix):
        target = raw / ('ris-' + prefix.replace('/', '_') + '.json')
        result, source = download(RIS + prefix, target)
        if result.get('status') != 'ok':
            raise ValueError(f'RIPEstat could not return BGP state for {prefix}: {result.get("messages")}')
        return prefix, result['data']['bgp_state'], source

    prefixes = sorted({p['prefix_v4'] for p in selected})
    with ThreadPoolExecutor(max_workers=4) as pool:
        routes = list(pool.map(fetch_routes, prefixes))
    by_prefix = {prefix: entries for prefix, entries, _ in routes}
    sample = []
    for p in selected:
        paths = set()
        for entry in by_prefix[p['prefix_v4']]:
            path = entry['path']
            if path and all(isinstance(asn, int) for asn in path) and path[-1] == p['asn_v4']:
                paths.add(tuple(dict.fromkeys(path)))
        if paths:
            shortest = min(paths, key=lambda path: (len(path), path))
            sample.append(dict(probe_id=p['id'], asn=p['asn_v4'], prefix=p['prefix_v4'],
                               country=p['country_code'], coordinates=p['geometry']['coordinates'],
                               paths=[shortest]))
    if len(sample) < 16:
        raise ValueError(f'Only {len(sample)} selected probes have matching routes; the simulation requires 16')
    manifest = dict(retrieved_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
                    selected=len(selected), retained=len(sample), raw_directory=raw.name,
                    inputs=[atlas_source] + [source for _, _, source in routes])
    for name, value in [('manifest.json', manifest), ('sample.json', sample)]:
        staged = raw / name
        staged.write_text(json.dumps(value, indent=2) + '\n')
        staged.replace(DATA / name)
    print(f'Collected routes for {len(sample)} of 50 selected RIPE Atlas probes in {DATA / "sample.json"}')


if __name__ == '__main__':
    collect()
