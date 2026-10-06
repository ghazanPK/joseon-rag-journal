"""Start this repository's bundled example; private data and model weights are optional.

By default the demo serves the bundled authored articles and works offline.
``--sample-crawl`` fetches a small Sejong Annals sample (one lunar month, at most
100 articles, about three minutes at 1.5 s per request; cached and resumable),
builds its index and serves it. ``--sample`` serves an already fetched sample
without touching the network. Neither runs unless asked.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CORPUS = 'data/sejong-sample.jsonl'
SAMPLE_INDEX = 'outputs/sejong-sample.index.json'
SAMPLE_QUESTION = '세종 2년 5월 살곶이 다리 공사를 감독한 사람은 누구인가?'
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--sample', action='store_true', help=f'serve the local Sejong sample ({SAMPLE_CORPUS}) instead of the authored examples; no network')
parser.add_argument('--sample-crawl', action='store_true', help='fetch the Sejong sample first (joseon-rag crawl --sample; cached reruns make no request), then serve it')
args = parser.parse_args()
os.chdir(ROOT)
os.environ['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')
prepare = ROOT / 'scripts' / 'prepare_viewer.py'
if prepare.exists() and not all((ROOT / 'static' / 'vendor' / name).is_file() for name in ('three.module.js', 'GLTFLoader.js', 'BufferGeometryUtils.js')):
    subprocess.run([sys.executable, str(prepare)], check=True)
subprocess.run([sys.executable, '-m', 'joseon_rag.cli', 'build', 'examples/articles.jsonl', '--out', 'outputs/demo.index.json'], check=True)


def sample_index(crawl: bool) -> str | None:
    """Build the sample index; return None (authored examples) when no sample corpus is available."""
    if crawl:
        print('Fetching the Sejong sample (one lunar month, at most 100 articles, 1.5 s between requests) ...', flush=True)
        if subprocess.run([sys.executable, '-m', 'joseon_rag.cli', 'crawl', '--sample', '--out', SAMPLE_CORPUS]).returncode:
            print('The sample crawl stopped (see the message above); rerun to resume from the cache.', flush=True)
    corpus = ROOT / SAMPLE_CORPUS
    if not corpus.is_file() or not corpus.read_text(encoding='utf-8').strip():
        print(f'No local sample at {SAMPLE_CORPUS}; serving the bundled authored examples. '
              'Fetch it with: python scripts/start_demo.py --sample-crawl', flush=True)
        return None
    subprocess.run([sys.executable, '-m', 'joseon_rag.cli', 'build', SAMPLE_CORPUS, '--out', SAMPLE_INDEX], check=True)
    return SAMPLE_INDEX


index = (sample_index(args.sample_crawl) if args.sample or args.sample_crawl else None) or 'outputs/demo.index.json'
extra = ['--question', SAMPLE_QUESTION] if index == SAMPLE_INDEX else []
if index != SAMPLE_INDEX and (ROOT / SAMPLE_CORPUS).is_file():
    print('A local Sejong sample exists; serve it with: python scripts/start_demo.py --sample', flush=True)
print(f'Open http://127.0.0.1:{args.port}/ — ' + ('the local Sejong Annals sample is ready (data stays on this machine).' if extra else 'bundled starter samples are ready.'), flush=True)
raise SystemExit(subprocess.call([sys.executable, *['-m', 'joseon_rag.cli', 'serve', index, '--events', 'examples/events.json', *extra], '--port', str(args.port)]))
