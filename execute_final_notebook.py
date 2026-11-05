"""Run the default notebook with genuine IPython outputs, requiring no TCP sockets."""
import argparse
import ast
from pathlib import Path
import platform
import sys
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parent


def require_saved_mode(notebook):
    values={}
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name) and target.id in ('MODE','ENABLE_FULL_REPRODUCTION'):
                        values[target.id]=ast.literal_eval(node.value)
    if values!={'MODE':'A','ENABLE_FULL_REPRODUCTION':False}:
        raise PermissionError('Headless packaging executor requires MODE="A" and ENABLE_FULL_REPRODUCTION=False')


def execute(output='final_experiment.executed.ipynb'):
    notebook=nbformat.read(ROOT/'final_experiment.ipynb',as_version=4);require_saved_mode(notebook)
    sys.path.insert(0,str(ROOT));shell=InteractiveShell.instance();count=0
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        count+=1
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result=shell.run_cell(cell.source,store_history=True)
        result.raise_error();outputs=[]
        if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for rich in captured.outputs:outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
        cell.outputs=outputs;cell.execution_count=count
        print(f'Executed saved-results cell {count}',flush=True)
    notebook.metadata.execution_engine='IPython in-process; no kernel sockets required'
    notebook.metadata.language_info=dict(name='python',version=platform.python_version())
    nbformat.validate(notebook)
    target=Path(output)
    if not target.is_absolute():target=ROOT/target
    if target.parent.resolve()!=ROOT:raise ValueError('Notebook output must stay in the project root')
    nbformat.write(notebook,target);return target

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='final_experiment.executed.ipynb')
    print(execute(parser.parse_args().output))
