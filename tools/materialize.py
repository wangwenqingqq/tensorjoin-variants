#!/usr/bin/env python3
"""Create a separate working copy with explicitly configured path placeholders."""
import argparse
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--map', action='append', default=[], metavar='TOKEN=VALUE',
                        help='Replace an additional path token, such as EXTERNAL_HOME=/opt/external')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    destination = args.destination.expanduser().resolve()
    if not re.fullmatch(r'/[A-Za-z0-9._/-]+', str(destination)):
        parser.error('destination must be an absolute portable path without spaces or shell metacharacters')
    if destination.exists():
        parser.error('destination must not already exist; existing files are never overwritten')
    if destination == source or source in destination.parents:
        parser.error('destination must be outside this archive')
    replacements = {'@TENSORJOIN_ROOT@': str(destination)}
    for item in args.map:
        token, sep, value = item.partition('=')
        if not sep or not re.fullmatch(r'[A-Z][A-Z0-9_]*', token) or not value:
            parser.error('--map must use TOKEN=VALUE with an uppercase token and nonempty value')
        if not re.fullmatch(r'/[A-Za-z0-9._/-]+', value):
            parser.error('mapped values must be absolute portable paths without spaces or shell metacharacters')
        if token == 'TENSORJOIN_ROOT':
            parser.error('TENSORJOIN_ROOT is always set from destination')
        replacements['@' + token + '@'] = value
    roots = sorted(p for p in source.iterdir() if p.is_dir() and p.name.startswith('tensorjoin_'))
    if not roots:
        parser.error('no TensorJoin archive directories found')
    destination.mkdir(parents=True)
    count = 0
    unresolved = set()
    for root in roots:
        target = destination / root.name
        shutil.copytree(root, target, symlinks=True)
        for path in target.rglob('*'):
            if not path.is_file() or path.is_symlink():
                continue
            data = path.read_bytes()
            try:
                text = data.decode('utf-8')
            except UnicodeError:
                continue
            if '\0' in text:
                continue
            updated = text
            for token, value in replacements.items():
                updated = updated.replace(token, value)
            unresolved.update(re.findall(r'@[A-Z][A-Z0-9_]*@', updated))
            if updated != text:
                path.chmod(path.stat().st_mode | 0o200)
                path.write_bytes(updated.encode('utf-8'))
                count += 1
    print(f'Created {len(roots)} project directories; replaced paths in {count} files.')
    if unresolved:
        print('Unresolved external paths: ' + ', '.join(sorted(unresolved)))
    print('No experiment, GPU command, dependency installation, or historical result was executed.')


if __name__ == '__main__':
    main()
