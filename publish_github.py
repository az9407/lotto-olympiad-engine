"""Create a public GitHub repository and publish an explicit artifact allowlist.

Credentials are read from environment, the GitHub CLI or Git's credential
helper in memory only. Tokens are never printed, embedded in URLs or saved.
No force push, reset, branch checkout or main merge is performed.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request


DEFAULT_BRANCH = 'feature/continuous-probability-engine'
SOURCE_NAMES = {
    '.gitignore', 'README.txt', 'README.md', 'requirements.txt', 'REVIEW.md',
    'predict_top15.py', 'rule_engine.py', 'number_model.py', 'publish_github.py',
    'lotto_analysis.py', 'combinatorial.py', 'exact_stats.py',
    'generate_candidates.py', 'rank_candidates.py',
}
ARTIFACT_NAMES = {
    'forecast_top15.csv', 'forecast_active_rules.csv', 'forecast_number_probabilities.csv',
    'forecast_report.html', 'top15.csv', 'active_rules.csv', 'number_probabilities.csv',
    'prediction_report.html', 'prediction_top15.csv', 'prediction_active_rules.csv',
    'prediction_number_probabilities.csv', 'prediction_report.txt',
    'report.html', 'eligible_active_rules.csv',
}


class PublishError(RuntimeError):
    pass


def _run(args, cwd=None, env=None, input_text=None, check=True):
    try:
        result = subprocess.run(args, cwd=cwd, env=env, input=input_text,
                                text=True, capture_output=True, encoding='utf-8', errors='replace', timeout=30)
    except subprocess.TimeoutExpired:
        raise PublishError(f'{Path(str(args[0])).name} {args[1]} timed out after 30 seconds') from None
    if check and result.returncode:
        # Git/CLI commands below never receive credential values as arguments.
        message = (result.stderr or result.stdout).strip()
        message = re.sub(r'(?:gh[opusr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)', '[REDACTED]', message)
        raise PublishError(f'{Path(str(args[0])).name} {args[1]} failed: {message[-1800:]}')
    return result


def find_token(root):
    for name in ('GH_TOKEN', 'GITHUB_TOKEN'):
        value = os.environ.get(name)
        if value:
            return value, f'environment:{name}'
    gh = shutil.which('gh')
    if not gh:
        candidates = [Path(os.environ.get('ProgramFiles', 'C:/Program Files'))/'GitHub CLI/gh.exe',
                      Path.home()/'AppData/Local/Microsoft/WinGet/Links/gh.exe']
        gh = next((str(p) for p in candidates if p.exists()), None)
    if gh:
        result = _run([gh, 'auth', 'token', '--hostname', 'github.com'], cwd=root, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip(), 'GitHub CLI'
    env = os.environ.copy()
    env.update({'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'never'})
    result = _run(['git', 'credential', 'fill'], cwd=root, env=env,
                  input_text='protocol=https\nhost=github.com\n\n', check=False)
    if result.returncode == 0:
        values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if values.get('password'):
            return values['password'], 'Git credential helper'
    raise PublishError('No usable GitHub authentication in GH_TOKEN/GITHUB_TOKEN, GitHub CLI or Git credential helper')


def api_request(token, method, path, payload=None):
    request = urllib.request.Request('https://api.github.com'+path, method=method,
        data=json.dumps(payload).encode('utf-8') if payload is not None else None,
        headers={'Authorization': 'Bearer '+token, 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2026-03-10', 'User-Agent': 'lotto-olympiad-engine',
                 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode('utf-8', errors='replace')
        try:
            message = json.loads(text).get('message', '')
        except ValueError:
            message = str(exc.reason)
        raise PublishError(f'GitHub API {method} {path}: HTTP {exc.code} {message}') from None
    except urllib.error.URLError as exc:
        raise PublishError(f'GitHub API network error: {exc.reason}') from None


def publication_files(root, additional=None):
    """Explicit source/report CSV/HTML allowlist; never stage source workbook or state."""
    files = []
    for path in root.iterdir():
        if path.is_file() and (path.name in SOURCE_NAMES or
                               (path.name.startswith('test_') and path.suffix == '.py')):
            files.append(path)
    output = root/'outputs'
    if output.exists():
        for path in output.iterdir():
            if path.is_file() and (path.name in ARTIFACT_NAMES or
                                   re.fullmatch(r'round\d+_(?:top15|active_rules|number_probabilities)\.csv', path.name) or
                                   re.fullmatch(r'round\d+_(?:forecast|prediction)_report\.html', path.name)):
                files.append(path)
    results = root/'results'
    if results.exists():
        for directory in sorted(results.iterdir()):
            if not directory.is_dir() or not re.fullmatch(r'round_\d+', directory.name):
                continue
            for name in ('top15.csv', 'active_rules.csv', 'eligible_active_rules.csv',
                         'number_probabilities.csv', 'report.html'):
                path = directory/name
                if path.is_file():
                    files.append(path)
    for raw in additional or []:
        path = (root/raw).resolve()
        if root not in path.parents or not path.is_file():
            raise PublishError(f'Additional publication path must be an existing workspace file: {raw}')
        if path.suffix.lower() not in {'.py', '.txt', '.md', '.csv', '.html'} or \
                any(part in {'.codex-attachments', '.git', '.aws', '.codex', '.cache', '.tooling'} for part in path.parts):
            raise PublishError(f'Additional publication path rejected: {raw}')
        files.append(path)
    return sorted({p.relative_to(root).as_posix() for p in files})


def prepare_repository(root, token, repo_name, branch):
    user = api_request(token, 'GET', '/user')
    owner = user['login']
    full_name = repo_name if '/' in repo_name else f'{owner}/{repo_name}'
    if full_name.split('/')[0].lower() != owner.lower():
        raise PublishError('Automatic creation is limited to the authenticated user account')
    created = False
    try:
        repository = api_request(token, 'GET', '/repos/'+full_name)
    except PublishError as exc:
        if 'HTTP 404 ' not in str(exc):
            raise
        repository = api_request(token, 'POST', '/user/repos',
            {'name': full_name.split('/')[1], 'private': False, 'auto_init': False,
             'description': 'Lotto mathematical feature and conditional transition scoring engine'})
        created = True
    url = repository['html_url']
    if not (root/'.git').exists():
        _run(['git', 'init', '-b', branch], cwd=root)
    current = _run(['git', 'symbolic-ref', '--short', 'HEAD'], cwd=root, check=False).stdout.strip()
    if current != branch:
        raise PublishError(f'Current branch is {current}; expected {branch}. No branch checkout was performed')
    remotes = _run(['git', 'remote'], cwd=root).stdout.split()
    if 'origin' in remotes:
        existing = _run(['git', 'remote', 'get-url', 'origin'], cwd=root).stdout.strip()
        if existing.rstrip('/').removesuffix('.git').lower() != url.lower():
            raise PublishError('Existing origin differs from requested repository; it was not overwritten')
    else:
        _run(['git', 'remote', 'add', 'origin', url+'.git'], cwd=root)
    for key, fallback in [('user.name', owner), ('user.email', f"{user['id']}+{owner}@users.noreply.github.com")]:
        if not _run(['git', 'config', '--get', key], cwd=root, check=False).stdout.strip():
            _run(['git', 'config', '--local', key, fallback], cwd=root)
    return {'repository': full_name, 'url': url, 'created': created,
            'branch': current, 'private': repository.get('private', False)}


def publish(root=None, repo='lotto-olympiad-engine', branch=DEFAULT_BRANCH,
            message='활성 규칙과 연속형 전이 점수를 결합한 다음 회차 후보 엔진 구현',
            prepare_only=False, dry_run=False, additional_files=None):
    root = Path(root or Path(__file__).parent).resolve()
    files = publication_files(root, additional_files)
    if dry_run:
        return {'status': 'dry-run', 'files': files, 'repository_name': repo, 'branch': branch}
    token, auth_source = find_token(root)
    result = prepare_repository(root, token, repo, branch)
    result['authentication_source'] = auth_source
    if prepare_only:
        result['status'] = 'prepared'
        return result
    if not files:
        raise PublishError('Publication allowlist is empty')
    # -f applies only to this explicit allowlist so output CSV/HTML can be
    # published while the rest of outputs (state, cache, old reports) stays ignored.
    _run(['git', 'add', '-f', '--', *files], cwd=root)
    staged = _run(['git', 'diff', '--cached', '--name-only'], cwd=root).stdout.splitlines()
    unexpected = set(staged)-set(files)
    if unexpected:
        raise PublishError('Other staged files exist; no commit was made: '+', '.join(sorted(unexpected)))
    if staged:
        _run(['git', 'commit', '-m', message], cwd=root)
    head = _run(['git', 'rev-parse', 'HEAD'], cwd=root).stdout.strip()
    # Use an in-memory per-process Git header without storing credentials or
    # placing them in the command line or remote URL.
    env = os.environ.copy()
    previous_count = int(env.get('GIT_CONFIG_COUNT', '0'))
    env['GIT_CONFIG_COUNT'] = str(previous_count+1)
    env[f'GIT_CONFIG_KEY_{previous_count}'] = 'http.https://github.com/.extraheader'
    env[f'GIT_CONFIG_VALUE_{previous_count}'] = 'AUTHORIZATION: basic '+base64.b64encode(('x-access-token:'+token).encode()).decode()
    env['GIT_TERMINAL_PROMPT'] = '0'
    _run(['git', 'push', '--set-upstream', 'origin', branch], cwd=root, env=env)
    remote = _run(['git', 'ls-remote', 'origin', f'refs/heads/{branch}'], cwd=root, env=env).stdout.strip().split()
    if not remote or remote[0] != head:
        raise PublishError('Remote commit verification failed after push')
    result.update({'status': 'pushed', 'commit': head, 'remote_verified': True, 'files': files,
                   'commit_url': result['url']+'/commit/'+head,
                   'branch_url': result['url']+'/tree/'+branch})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='lotto-olympiad-engine')
    parser.add_argument('--branch', default=DEFAULT_BRANCH)
    parser.add_argument('--message', default='활성 규칙과 연속형 전이 점수를 결합한 다음 회차 후보 엔진 구현')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--include', action='append', default=[])
    args = parser.parse_args()
    try:
        result = publish(repo=args.repo, branch=args.branch, message=args.message,
                         prepare_only=args.prepare_only, dry_run=args.dry_run, additional_files=args.include)
    except (PublishError, OSError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc),
                          'proposed_repository': 'https://github.com/az9407/'+args.repo.rsplit('/', 1)[-1]}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
