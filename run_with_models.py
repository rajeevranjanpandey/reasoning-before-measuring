"""Delivery launcher added 10 October 2026; recorded research scripts are unchanged."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import ssl
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
MODELS = {
    'qwen': (
        'Qwen3-1.7B-Q8_0.gguf',
        'https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/90862c4b9d2787eaed51d12237eafdfe7c5f6077/Qwen3-1.7B-Q8_0.gguf',
        '061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a',
        1834426016,
    ),
    'granite': (
        'granite-3.3-2b-instruct-Q8_0.gguf',
        'https://huggingface.co/ibm-granite/granite-3.3-2b-instruct-GGUF/resolve/7cdf86ccd1f1bb3491c9b7017b033f2e51367397/granite-3.3-2b-instruct-Q8_0.gguf',
        '39068a5bcb15f10487a49426d617b1b0f38a624d7dbdefe5ff6f26de82d97039',
        2694120096,
    ),
}


def tls_context():
    context = ssl.create_default_context()
    # Python.org macOS installations may lack their own certificate bundle.
    if sys.platform == 'darwin' and Path('/etc/ssl/cert.pem').is_file():
        context.load_verify_locations('/etc/ssl/cert.pem')
    return context


def verified(path, checksum, size):
    if path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == checksum


def ensure_file(target, url, checksum, size):
    target = Path(target)
    if target.exists():
        if not verified(target, checksum, size):
            raise ValueError('Existing model fails verification; preserved: ' + str(target))
        print('Verified cached model:', target.name, flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.part', delete=False) as output:
            partial = Path(output.name)
            print('Downloading:', target.name, flush=True)
            with urllib.request.urlopen(url, timeout=60, context=tls_context()) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(chunk)
        if not verified(partial, checksum, size):
            raise ValueError('Downloaded model fails size/SHA-256 verification: ' + target.name)
        if target.exists():
            if not verified(target, checksum, size):
                raise ValueError('Another process created an invalid model; preserved: ' + str(target))
        else:
            os.replace(partial, target)
        print('Verified model:', target.name, flush=True)
    finally:
        if partial is not None and partial.exists():
            partial.unlink()


def main():
    parser = argparse.ArgumentParser(description='Restore pinned weights, then run an unchanged research script. Without a script, restore weights only.')
    parser.add_argument('--model', choices=['qwen', 'granite', 'both'], default='both')
    parser.add_argument('script', nargs='?')
    parser.add_argument('script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = None
    if args.script:
        script = (ROOT / args.script).resolve()
        if (ROOT / 'src').resolve() not in script.parents or script.suffix != '.py' or not script.is_file():
            parser.error('Choose an existing Python script inside this package\'s src folder.')
    try:
        for model in (MODELS if args.model == 'both' else [args.model]):
            filename, url, checksum, size = MODELS[model]
            ensure_file(ROOT / 'models' / filename, url, checksum, size)
    except (OSError, ValueError) as error:
        print('Model restoration failed:', error, file=sys.stderr)
        return 1
    if script:
        return subprocess.run([sys.executable, str(script), *args.script_args], cwd=ROOT).returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
