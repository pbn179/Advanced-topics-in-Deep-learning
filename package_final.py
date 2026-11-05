"""Build the complete research ZIP without training, installing or touching results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from final_project import ROOT,digest

ZIP_NAME='lottery_ticket_research_final.zip'
PREFIX='lottery_ticket_research_final'


def preserve_originals(root=ROOT):
    root=Path(root)
    hashes=json.loads((root/'packaging/original_artifact_checksums.json').read_text())
    for name,h in hashes.items():
        if not (root/name).is_file() or digest(root/name)!=h:raise ValueError(f'Original research artifact changed: {name}')
    return len(hashes)


def plan(root=ROOT):
    root=Path(root);selected={};aliases={};excluded=[]
    notebooks={digest(p):p for p in (root/'archive/notebooks').glob('*.ipynb')}
    for directory in ('reports','results','tests','archive','paper_assets','packaging'):
        for p in (root/directory).rglob('*'):
            if not p.is_file():continue
            name=p.relative_to(root).as_posix()
            if any(x in p.parts for x in ('__pycache__','.pytest_cache','.ipynb_checkpoints')) or p.suffix in ('.pyc','.log','.lock','.tmp'):
                excluded.append(name);continue
            if directory=='packaging' and p.name in ('archive_contents.json','artifact_manifest.json','release_manifest.json'):continue
            if p.suffix=='.ipynb' and directory in ('reports','results'):
                h=digest(p)
                if h in notebooks:
                    aliases[name]=notebooks[h].relative_to(root).as_posix();continue
                archived=root/'archive/notebooks'/('historical__'+name.replace('/','__'))
                if not archived.exists():shutil.copy2(p,archived)
                notebooks[h]=archived;selected[archived.relative_to(root).as_posix()]=archived
                aliases[name]=archived.relative_to(root).as_posix();continue
            # Exclude a live history only when the final CSV is byte-identical.
            canonical=None
            if p.name=='history_live.csv':canonical=p.with_name('history.csv')
            elif p.name.endswith('_live_history.csv'):
                canonical=p.with_name('history_'+p.name.removesuffix('_live_history.csv')+'.csv')
            if canonical and canonical.is_file() and digest(canonical)==digest(p):
                aliases[name]=canonical.relative_to(root).as_posix();continue
            selected[name]=p
    for p in root.iterdir():
        if p.is_file() and (p.suffix in ('.py','.yaml','.pdf') or p.name=='README.md' or p.name.startswith('requirements') and p.suffix=='.txt' or p.name=='final_experiment.ipynb'):
            selected[p.name]=p
    return selected,aliases,excluded


def prepare_manifest(root=ROOT):
    root=Path(root);(root/'packaging').mkdir(exist_ok=True)
    files,aliases,excluded=plan(root)
    # Stable data-only manifest avoids circular hashes of executed notebooks/release reports.
    data={n:digest(p) for n,p in files.items() if n.startswith(('reports/','results/','archive/'))}
    (root/'packaging/artifact_manifest.json').write_text(json.dumps(data,indent=2))
    (root/'packaging/archive_contents.json').write_text(json.dumps(dict(
        canonical_files=len(files),original_notebook_and_live_history_aliases=aliases,
        excluded_operational_files=sorted(excluded),excluded_directories=['.git','.venv','datasets','__pycache__','.pytest_cache','.ipynb_checkpoints','reproductions'],
        all_model_checkpoints_included=True,checkpoint_archives_required=1),indent=2))
    return len(data)


def build(root=ROOT):
    root=Path(root);originals=preserve_originals(root);prepare_manifest(root)
    files,aliases,excluded=plan(root)
    for name in ('packaging/artifact_manifest.json','packaging/archive_contents.json'):files[name]=root/name
    release={name:digest(p) for name,p in sorted(files.items())}
    (root/'packaging/release_manifest.json').write_text(json.dumps(release,indent=2))
    files['packaging/release_manifest.json']=root/'packaging/release_manifest.json'
    archive=root/ZIP_NAME
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,p in sorted(files.items()):z.write(p,f'{PREFIX}/{name}')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC verification failed')
    info=dict(file_count=len(files),checkpoint_files=sum(p.suffix=='.pt' for p in files.values()),
        uncompressed_bytes=sum(p.stat().st_size for p in files.values()),zip_bytes=archive.stat().st_size,
        sha256=digest(archive),previous_files_unchanged=originals,all_checkpoints_in_main_archive=True)
    # External sidecar avoids a ZIP self-hash cycle.
    (root/'lottery_ticket_research_final.zip.sha256').write_text(info['sha256']+'  '+ZIP_NAME+'\n')
    print(json.dumps(info,indent=2));return info

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest-only',action='store_true');args=parser.parse_args()
    print(prepare_manifest()) if args.manifest_only else build()
