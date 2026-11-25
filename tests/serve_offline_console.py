"""UI-only preview with Linux configuration; hardware actions are disabled."""
import configparser
import os
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='dobot-ui-') as directory:
        config=configparser.ConfigParser()
        config.read(ROOT/'config/linux.example.ini',encoding='utf-8')
        for side in ('LEFT','RIGHT'):
            config['GRIPPER_'+side]['port']='/dev/dobot-gripper-'+side.lower()
        path=Path(directory)/'linux.ini'
        with path.open('w',encoding='utf-8') as f:config.write(f)
        os.environ['DOBOT_CONFIG']=str(path)
        from tools import local_console as app
        original=app.commands
        def commands(project=app.DEFAULT_PROJECT,episode=''):
            from tools.linux_console import commands as linux
            return linux(project,episode,ROOT,app.PYTHON,app.LOCK)
        app.commands=commands
        def refuse(*args,**kwargs):raise ValueError('Offline preview: hardware actions disabled')
        app.Console.submit=refuse
        # Present Linux labels even when previewing on Windows.
        original_reply=app.Handler.reply
        def reply(self,status,data,*args,**kwargs):
            if isinstance(data,dict) and 'endpoints' in data:
                data.update(shell='Bash',data_root='/workspace/datasets',separator='/')
            return original_reply(self,status,data,*args,**kwargs)
        app.Handler.reply=reply
        app.main()
