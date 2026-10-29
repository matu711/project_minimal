"""Foreground launcher for the unchanged train.sh, with console and GPU logs.

Run inside the existing tuning container. No detach/nohup/background training.
The parent stays attached, waits for training, and Ctrl-C reaches the child.
"""
import argparse,datetime,os,pathlib,subprocess,sys,threading
import yaml
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=pathlib.Path);a=p.parse_args()
config=a.config.resolve();params=yaml.safe_load(config.read_text());name=f"{params['model']}_piper_{params['robot']}_{params['method']}";folder=ROOT/'logs'/name/params['exp_name'];folder.mkdir(parents=True,exist_ok=True)
phase='resume' if params.get('resume') else 'train';stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');console=folder/f'console_{phase}_{stamp}.log';gpu=folder/f'gpu_{phase}_{stamp}.csv';stop=threading.Event()
def monitor():
 with gpu.open('x',buffering=1) as f:
  f.write('utc_time,index,memory_used_mib,memory_free_mib,gpu_util_percent\n')
  while not stop.is_set():
   r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
   for line in r.stdout.splitlines():f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+','+line+'\n')
   stop.wait(1)
thread=threading.Thread(target=monitor,daemon=True);thread.start();print('Foreground training; console:',console,'GPU:',gpu,flush=True)
import signal
try:
 with console.open('x') as output:
  child=subprocess.Popen(['sh',str(ROOT/'scripts/train.sh'),'--config',str(config)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  def tee():
   for line in child.stdout:
    sys.stdout.write(line);sys.stdout.flush();output.write(line);output.flush()
  reader=threading.Thread(target=tee,daemon=True);reader.start()
  while True:
   try:code=child.wait();break
   except KeyboardInterrupt:
    # Stay in the foreground until our child exits; never orphan training.
    if child.poll() is None:child.send_signal(signal.SIGINT)
  reader.join()
finally:stop.set();thread.join(timeout=3)
sys.exit(code if code>=0 else 128-code)
