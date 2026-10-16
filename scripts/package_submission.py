"""Package an actual clean Git clone plus supplied FD001 data for submission.

Usage: python scripts/package_submission.py --clone PATH --data-dir PATH --zip PATH
The source clone must have a clean tracked working tree and no unexpected files.
This creates an archive, never deletes or changes the source repository/data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clone', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--zip', type=Path, required=True)
    args = parser.parse_args()
    clone = args.clone.resolve()
    if not (clone/'.git').is_dir():
        raise ValueError('A real Git clone including its .git directory is required')
    status = subprocess.check_output(['git','status','--porcelain'],cwd=clone,text=True)
    if status.strip():
        raise ValueError('Clone must be clean before packaging')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=clone,text=True).strip()
    manifest = json.loads((clone/'data/manifest.json').read_text(encoding='utf-8'))
    for name, record in manifest['files'].items():
        if hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Data hash mismatch: {name}')
    if args.zip.exists():
        raise FileExistsError('Choose a new ZIP name to preserve existing submission archives')
    args.zip.parent.mkdir(parents=True,exist_ok=True)
    prefix = 'MMA3001Project'
    with ZipFile(args.zip,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        for source in sorted(clone.rglob('*')):
            relative = source.relative_to(clone)
            if source.is_file() and not any(p in {'.private','.venv','__pycache__','.pytest_cache','node_modules'} for p in relative.parts):
                archive.write(source, f'{prefix}/{relative.as_posix()}')
        for name in manifest['files']:
            archive.write(args.data_dir/name, f'{prefix}/data/raw/{name}')
        archive.writestr(f'{prefix}/SUBMISSION_SNAPSHOT.txt',
            f'Git commit: {commit}\nIncludes Git history and the supplied FD001 files for local assessment reproduction.\n'
            'NASA data are separate from the project MIT license. Follow README setup instructions.\n'
            'Complete personal reflection and rehearsal before assessment.\n')
    with ZipFile(args.zip) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
        names=set(archive.namelist())
        required=['.git/HEAD','README.md','data/raw/train_FD001.txt','data/raw/test_FD001.txt',
                  'data/raw/RUL_FD001.txt','output/pdf/Jet_Engine_RUL_Report.pdf',
                  'presentation/Jet_Engine_RUL_Presentation.pptx']
        if any(f'{prefix}/{name}' not in names for name in required):
            raise ValueError('Archive is missing a required deliverable')
    print(json.dumps({'zip':str(args.zip.resolve()),'commit':commit,'bytes':args.zip.stat().st_size,
                      'sha256':hashlib.sha256(args.zip.read_bytes()).hexdigest(),'verified':True},indent=2))


if __name__ == '__main__':
    main()
