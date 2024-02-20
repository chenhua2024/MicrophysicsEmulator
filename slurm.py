import os,sys
import socket
import subprocess as sp
import pandas as pd

host_name = socket.gethostname()
if 'Orion'  in host_name:
    username = 'hleighto'
else:
    username = 'Hua.Leighton'

def get_batch_head(partition=None, errname='errname',outname='outname',time=None,
                   jobname='jobname',account=None,bigmem=None,qos=None,
                   nodes=None,ntasks=None,cmd=None):
    params = [
            '#!/bin/sh',
            f'#SBATCH -e {errname}',
            f'#SBATCH -o {outname}',
            f'#SBATCH --job-name={jobname}']
    if account:
        params.append(f'#SBATCH --account={account}')
    if qos:
        params.append(f'#SBATCH --qos={qos}')
    if partition:
        params.append(f'#SBATCH --partition={partition}')
    if nodes:
        params.append(f'#SBATCH --nodes={nodes}')
    if ntasks:
        params.append(f'#SBATCH --ntasks-per-node={ntasks}')
    if time:
        params.append(f'#SBATCH --time={time}')
    if bigmem:
        params.append(f'#SBATCH --mem-per-cpu={bigmem}G')
    if cmd:
        params.append(cmd)
    return '\n'.join(params)    

def create_batchfile(task, filename,**kwargs):
    with open(filename, 'w') as f:
        f.write(get_batch_head(**kwargs))
        f.write('\n')
        f.write(task)

def get_njobs(username=username,**kwargs):
    cmd=f'squeue -u {username}'
    output=sp.check_output(cmd,shell=True).decode('utf-8').split('\n')[:-1]
    if not kwargs:
        return len(output)-1
    else:
        if 'jobname' in kwargs.keys():
            njobs = [line for line in output if kwargs['jobname'] in line]
            return len(njobs)
        
def sc_jobs(**kwargs):
    cmd=f'squeue -u {username}'
    lines = sp.check_output(cmd,shell=True).decode('utf8').split('\n')[1:-1]
    df = pd.DataFrame(data=[[line.split()[0],line.split()[2],line.split()[4]] for line in lines],columns=['jobid','jobname','jobstatus'])
    if not kwargs:
        [os.system(f'scancel {jobid}') for jobid in df['jobid']]
    else:
        if 'jobname' in kwargs.keys() and 'jobstatus' not in kwargs.keys():
            df_sub = df.loc[df['jobname']==kwargs['jobname']]
        elif 'jobstatus' in kwargs.keys() and 'jobname' not in kwargs.keys():
            df_sub = df.loc[df['jobstatus']==kwargs['jobstatus']]
        elif 'jobstatus' in kwargs.keys() and 'jobname' in kwargs.keys():
            df_sub = df.loc[(df['jobstatus']==kwargs['jobstatus']) & (df['jobname']==kwargs['jobname'])]
        elif 'number' in kwargs.keys():
            df_sub = df.iloc[:kwargs['number']]
        [os.system(f'scancel {jobid}') for jobid in df_sub['jobid']]


def get_slurmfiles_in_queue(username=username):
    cmd=f'squeue -u {username}'
    job_ids = [line.split()[0] for line in sp.check_output(cmd,shell=True).decode('utf8').split('\n')[1:-1]]
    slurm_files = []
    for job_id in job_ids:
        cmd=f'scontrol show jobid {job_id}'
        slurm_files.extend(line.rstrip('/').split('/')[-1] for line in sp.check_output(cmd,shell=True).decode('utf8').split('\n') if 'Command=' in line)
    return slurm_files

