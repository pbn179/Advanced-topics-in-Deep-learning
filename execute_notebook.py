"""Execute saved-results notebook cells with genuine IPython outputs, no TCP sockets."""
import argparse,ast,platform,sys
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
ROOT=Path(__file__).resolve().parent


def require_saved(notebook):
    values={}
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node,ast.Assign):
                for t in node.targets:
                    if isinstance(t,ast.Name) and t.id in ('MODE','ENABLE_TRAINING'):values[t.id]=ast.literal_eval(node.value)
    if values!={'MODE':'saved','ENABLE_TRAINING':False}:raise PermissionError('Packaging execution only accepts saved mode with training disabled')


def execute(output):
    n=nbformat.read(ROOT/'final_experiment.ipynb',as_version=4);require_saved(n)
    sys.path.insert(0,str(ROOT));shell=InteractiveShell.instance();count=0
    for c in n.cells:
        if c.cell_type!='code':continue
        count+=1
        with capture_output(stdout=True,stderr=True,display=True) as cap:result=shell.run_cell(c.source,store_history=True)
        result.raise_error();outputs=[]
        if cap.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=cap.stdout))
        if cap.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=cap.stderr))
        for o in cap.outputs:outputs.append(nbformat.v4.new_output('display_data',data=o.data,metadata=o.metadata))
        c.outputs=outputs;c.execution_count=count;print(f'Executed saved-results cell {count}',flush=True)
    n.metadata.execution_engine='IPython in-process';n.metadata.language_info=dict(name='python',version=platform.python_version())
    nbformat.validate(n);target=Path(output)
    if not target.is_absolute():target=ROOT/target
    if target.parent.resolve()!=ROOT:raise ValueError('Save executed notebooks in this project root')
    nbformat.write(n,target);return target

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='final_experiment.executed.ipynb');a=p.parse_args();print(execute(a.output))
