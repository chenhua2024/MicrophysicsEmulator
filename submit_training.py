# from slurm import create_batchfile
import os,sys
import numpy as np
import subprocess as sp
from slurm import create_batchfile

bs=2048


python_sub_script = sys.argv[1]
exp_data = sys.argv[2]
nb = int(sys.argv[3])
lr = float(sys.argv[4])

exe = '/contrib/Hua.Leighton/software/miniconda/bin/python'
mp_ai_root = '/contrib/Hua.Leighton/mp_ai'


python_scripts_path = os.path.join(mp_ai_root,'code/resnn2')
slurm_scripts_root = os.path.join(mp_ai_root,'slurm_scripts/training')
cron_outputs_root = os.path.join(mp_ai_root,'cron_outputs/training')

print(python_sub_script,exp_data)
exp_name = f"{python_sub_script.replace('cloud_train_','').replace('.py','')}_{exp_data}"
slurm_scripts_path = os.path.join(slurm_scripts_root,exp_name)
cron_outputs_path = os.path.join(cron_outputs_root,exp_name)
os.makedirs(slurm_scripts_path,exist_ok=True)
os.makedirs(cron_outputs_path,exist_ok=True)

os.chdir(cron_outputs_path)

params = f'nb={nb},lr={lr},bs={bs}'
cmd=f'{exe} {python_scripts_path}/{python_sub_script} {exp_data} {nb} {lr}'
slurm_file=os.path.join(slurm_scripts_path,exp_name+'_'+params.replace(',','_').replace('=','')+'.sh')

create_batchfile(cmd,slurm_file,
            errname=f"{exp_name}_{params.replace(',','_').replace('=','')}.err", 
            outname=f"{exp_name}_{params.replace(',','_').replace('=','')}.out",jobname=exp_name)                               
os.system(f'sbatch --mem=0 {slurm_file}')  
print(f'sbatch {slurm_file}')


