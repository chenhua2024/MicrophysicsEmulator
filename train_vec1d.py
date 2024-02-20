from netCDF4 import Dataset
import numpy as np
import os,sys,glob
import subprocess as sp

import tensorflow as tf
from keras.optimizers import Adam
from resmp import ResCu
import tensorflow as tf

def get_path():
    mp_ai_root = '/contrib/Hua.Leighton/mp_ai'    
    saved_json_path = os.path.join(mp_ai_root,'saved_json',exp_name)
    saved_weights_path = os.path.join(mp_ai_root,'saved_weights',exp_name)
    loss_path = os.path.join(mp_ai_root,'losses',exp_name)
    training_path = os.path.join(mp_ai_root,'training_data',exp_data)
    os.makedirs(saved_json_path,exist_ok=True)
    os.makedirs(saved_weights_path,exist_ok=True)
    os.makedirs(loss_path,exist_ok=True)    
    return loss_path,saved_json_path,saved_weights_path,training_path

def build_model():
    model = ResCu((81,14,81,10),nb_residual_unit = nb)
    adam = Adam(learning_rate=lr,decay=1e-6)
    model.compile(loss='mean_squared_error', optimizer=adam)    
    return model

def json_loss_files():
### save model architecture
    json_file = os.path.join(saved_json_path,f'{exp_name}_nb{nb}_lr{lr}_bs{batch_size}.json')
    open(json_file,'w').write(model.to_json())

    ### create loss file to save losses during training
    loss_file = os.path.join(loss_path,f'{exp_name}_nb{nb}_lr{lr}_bs{batch_size}.txt')
    open(loss_file,'w').close()    
    return json_file,loss_file

def get_training_files():
    os.chdir(training_path)
    return sp.check_output('find . -name "*.nc"',shell=True).decode('utf8').split('\n')[:-1]    

script_name = sys.argv[0]
exp_data = sys.argv[1]
exp_name = f"{os.path.basename(script_name).replace('cloud_train_','').replace('.py','')}_{exp_data}"
if len(sys.argv) == 4:
    nb = int(sys.argv[2])
    lr = float(sys.argv[3])
else:
    nb = 3
    lr = 0.0003

ai_vars = {0: 'qv_in', 1: 'qc_in', 2: 'qi_in', 3: 'qr_in', 4: 'qs_in', 
           5: 'qg_in', 6: 'ni_in', 7: 'nr_in', 8: 't_in', 9: 'pfils_in',
           10: 'pflls_in', 11: 'p_in', 12: 'w_in', 13: 'dz_in', 14: 'qv_tend', 
           15: 'qc_tend', 16: 'qi_tend', 17: 'qr_tend', 18: 'qs_tend', 
           19: 'qg_tend', 20: 'nr_tend', 21: 't_tend', 22: 'pfils_tend', 
           23: 'pflls_tend', 24: 'pptrain_out', 25: 'pptsnow_out', 26: 'pptgraul_out', 
           27: 'pptice_out', 28: 'ni_tend'}


loss_path,saved_json_path,saved_weights_path,training_path = get_path()
batch_size = 2048
nepoch = 100
model = build_model()
json_file,loss_file = json_loss_files()
files = get_training_files()

print(f'exp_data: {exp_data}')
print(f'exp_name: {exp_name}')
print(f'training_path: {training_path}')
for iepoch in range(nepoch):
    print(iepoch)
    for file in files:
        fh = Dataset(file,'r')   
        ary = fh.variables['ai_vars'][:,:,:]
        inputs = fh.variables['ai_vars'][:,:,:14]
        outputs= fh.variables['ai_vars'][:,:,14:24]
        history = model.fit( inputs , outputs , epochs=1, verbose=2, batch_size=batch_size)
        os.system(f"echo {iepoch} {os.path.basename(file).replace('.nc','')} {history.history['loss'][0]} >> {loss_file}")
        fh.close()
    model.save_weights(os.path.join(saved_weights_path,f"{exp_name}_nb{nb}_lr{lr}_bs{batch_size}_epoch{iepoch}.h5"))






