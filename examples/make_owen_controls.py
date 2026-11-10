"""Regenerate original control requests and shared wording, not numerical reports."""
import json
from pathlib import Path
from shapleylab.owen_contract import SCOPE,REPORT_SCOPE
from shapleylab.owen_controls import definitions


def main():
    root=Path(__file__).resolve().parents[1]
    data={'wording':{'SCOPE':SCOPE,'REPORT_SCOPE':REPORT_SCOPE},'controls':definitions()}
    text='globalThis.OWEN_WORDING='+json.dumps(data['wording'],ensure_ascii=False,separators=(',',':'))+';\nglobalThis.OWEN_CONTROLS='+json.dumps(data['controls'],ensure_ascii=False,separators=(',',':'))+';\n'
    (root/'docs/owen-data.js').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
