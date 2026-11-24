"""Fetch the current Validators.app hosting-ASN table and plot its stake-share capacity screen."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.request import urlopen

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'/'asn'
URL = 'https://www.validators.app/data-centers?locale=en&network=mainnet&sort_by=asn'


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell, self.skip = [], None, None, False

    def handle_starttag(self, tag, attrs):
        if tag in ['script','style']:
            self.skip = True
        elif tag == 'tr':
            self.row = []
        elif tag in ['td','th'] and self.row is not None:
            self.cell = []

    def handle_data(self, value):
        if self.cell is not None and not self.skip:
            self.cell.append(value)

    def handle_endtag(self, tag):
        if tag in ['script','style']:
            self.skip = False
        elif tag in ['td','th'] and self.cell is not None:
            self.row.append(' '.join(' '.join(self.cell).split()))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def parse_source_table(content):
    table = TableParser()
    table.feed(content.decode('utf-8'))
    header = next((r for r in table.rows if r and r[0]=='ASN'), None)
    if header is None:
        raise ValueError('The Validators.app response contains no hosting-ASN table')
    population = int(re.search(r'\(([\d,]+)\)', header[3])[1].replace(',',''))
    total_stake = int(re.search(r'\(([\d,]+)\)', header[5])[1].replace(',',''))
    rows = []
    for row in table.rows:
        if len(row)!=7 or not re.match(r'^\d+(?:\s|$)',row[0]):
            continue
        match = re.fullmatch(r'(\d+)(?:\s+(.*))?',row[0])
        stake_match = re.fullmatch(r'([\d,]+)\s+\(([\d.]+)%\)',row[5])
        assert match and stake_match, row[0]
        stake = int(stake_match[1].replace(',',''))
        share = stake/total_stake
        displayed_percent = float(stake_match[2])
        assert abs(100*share-displayed_percent)<=.011, row[0]
        rows.append(dict(asn=int(match[1]), organization=match[2] or 'Unclassified',
                         validators=int(row[3].replace(',','')), displayed_active_stake_sol=stake,
                         share=share, displayed_percent=displayed_percent))
    return population, total_stake, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-html', type=Path, help='Analyze a copy of the public page that you saved')
    args = parser.parse_args()
    if args.source_html:
        content = args.source_html.read_bytes()
        retrieved = datetime.fromtimestamp(args.source_html.stat().st_mtime, timezone.utc)
        method = 'browser-saved HTML; timestamp from file modification time'
    else:
        try:
            with urlopen(URL, timeout=30) as response:
                content = response.read()
        except HTTPError as error:
            if error.code == 403:
                parser.exit(1, f'Direct download refused. Open {URL} in your browser, save the HTML, '
                               'and rerun with --source-html PATH.\n')
            raise
        retrieved = datetime.now(timezone.utc)
        method = 'direct HTTP retrieval'
    population, total_stake, rows = parse_source_table(content)
    manifest = dict(url=URL, retrieved_at_utc=retrieved.isoformat(), acquisition=method,
                    sha256=hashlib.sha256(content).hexdigest(), source='Validators.app mainnet ASN aggregation',
                    unknown_rule='ASN 0 included in stake denominator, excluded from ASN bars')
    assert len({r['asn'] for r in rows})==len(rows)
    assert sum(r['validators'] for r in rows)==population
    # The page displays whole SOL; independent rounding of rows may differ from its total.
    discrepancy = sum(r['displayed_active_stake_sol'] for r in rows)-total_stake
    assert abs(discrepancy)<=len(rows), discrepancy
    known = sorted([r for r in rows if r['asn']!=0], key=lambda r:r['displayed_active_stake_sol'],reverse=True)
    unknown = [r for r in rows if r['asn']==0]
    largest = known[0]
    largest_share = Fraction(largest['displayed_active_stake_sol'],total_stake)
    screen = []
    for N in range(2,9):
        ratio = N*largest_share
        minimum_budget = (ratio.numerator+ratio.denominator-1)//ratio.denominator
        screen.append(dict(N=N, one_per_asn_threshold=1/N,
                           known_asns_pass_stake_share_screen=largest_share<=Fraction(1,N),
                           largest_asn_necessary_budget_under_proportional_quotas=minimum_budget))
    result = dict(source=manifest, listed_validators=population, known_asns=len(known),
                  displayed_total_active_stake_sol=total_stake,
                  row_sum_minus_total_sol=discrepancy, unclassified=unknown,
                  largest_asn=largest, rows=rows, idealized_screen=screen,
                  interpretation='Stake-share screening under exact proportional quotas; no realized MCP schedule or feasibility result for combined policies')
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA/'source.html').write_bytes(content)
    (DATA/'snapshot.json').write_text(json.dumps(result,indent=2)+'\n')

    top = known[:7]
    labels = [f'{r["organization"].split(",")[0]}  AS{r["asn"]}' for r in top]
    shares = [100*r['share'] for r in top]
    fig = plt.figure(figsize=(6.4, 3.9))
    ax = fig.add_axes([.34, .15, .64, .63])
    fig.text(.025, .96, 'Hosting-ASN stake shares and one-proposer-per-ASN limits',
             fontsize=11, va='top')
    ax.barh(range(len(top)), shares, color=['#b27813']+['#187c92']*(len(top)-1),
            height=.62, zorder=3)
    ax.set_yticks(range(len(top)), labels, fontsize=10)
    ax.set_ylim(len(top)-.4, -.6)
    for y, value in enumerate(shares):
        if y == 0:
            ax.text(value-.7, y, f'{value:.2f}%', ha='right', va='center',
                    fontsize=10, color='white', fontweight='bold', zorder=4)
        else:
            ax.text(value+.7, y, f'{value:.2f}%', va='center', fontsize=10)
    for N, label_x in [(5,13), (4,25), (3,37), (2,50)]:
        limit = 100/N
        ax.axvline(limit, color='#8a8a8a', linestyle='--', linewidth=.9, zorder=1)
        ax.annotate(f'{N} proposers\n{limit:.1f}%', xy=(limit,1),
                    xycoords=ax.get_xaxis_transform(), xytext=(label_x,1.07),
                    textcoords=ax.get_xaxis_transform(), ha='center', va='bottom',
                    fontsize=9.5, linespacing=1.25,
                    arrowprops=dict(arrowstyle='-', color='#8a8a8a', linewidth=.8))
    upper_tick = max(50, 10 * math.ceil(max(shares) / 10))
    ax.set_xlim(0, upper_tick + 5)
    ticks = range(0, upper_tick + 1, 10)
    ax.set_xticks(ticks, [f'{tick}%' for tick in ticks])
    ax.set_xlabel('Share of reported stake', fontsize=11, labelpad=8)
    ax.tick_params(axis='x',labelsize=10)
    ax.tick_params(axis='y',length=0)
    ax.spines[['top','right','left']].set_visible(False)
    figures = ROOT/'figures'
    figures.mkdir(exist_ok=True)
    fig.savefig(figures/'asn-capacity.pdf', bbox_inches='tight')
    fig.savefig(figures/'asn-capacity.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({k:result[k] for k in ['listed_validators','known_asns','row_sum_minus_total_sol',
                                         'largest_asn','unclassified','idealized_screen']},indent=2))


if __name__=='__main__':
    main()
