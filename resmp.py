import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D, 
    Dense,
    BatchNormalization,
    Activation,
    add,
    Add)
from tensorflow.keras.models import Model
import sys


def resmp(nz_in,nChannel_in,nz_out,nChannel_out,bottom_layer_dict=None,top_layer_dict=None,res_dict=None,nblocks=3):
    if not bottom_layer_dict:
        bottom_layer_dict = {'filters':128, 'kernel_size':3, 'activation':'relu'}
    if not top_layer_dict:
        top_layer_dict = {'kernel_size':3, 'activation':'tanh'}    
    if not res_dict:
        res_dict = {
            'nlayers': 3,
            'layer info': [
            {'filters':128, 'kernel_size':3,'activation':'relu'}
            ],
            'last activation':'relu',
        }

    nlayers = res_dict['nlayers']
    nlayers_provided = len(res_dict['layer info'])
    if nlayers == nlayers_provided:
        res_dict['layer info'][-1]['activation']=None
    elif nlayers > nlayers_provided:
        filters = res_dict['layer info'][-1]['filters']
        kernel_size = res_dict['layer info'][-1]['kernel_size']       
        activation =  res_dict['layer info'][-1]['activation']  
        for _ in range(nlayers-nlayers_provided-1):
            res_dict['layer info'].append({'filters': filters, 'kernel_size': kernel_size, 'activation': activation})            
        res_dict['layer info'].append({'filters': filters, 'kernel_size': kernel_size, 'activation': None})    
    else:
        sys.exit('provided layer parameters are greater than desired nlayers')
 
    x = Input(shape=(nz_in,nChannel_in))
    x = Conv1D(bottom_layer_dict['filters'], bottom_layer_dict['kernel_size'],padding='same')(x)
    # for _ in range(nblocks):
    for val in res_dict['layer info']:
        filters = val['filters']
        kernel_size = val['kernel_size']
        activation = val['activation']
        inputs = x
        x = Conv1D(filters, kernel_size,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)    
    x = add([x, inputs])
    x = Activation(res_dict['last activation'])(x)

    for val in res_dict['layer info']:
        filters = val['filters']
        kernel_size = val['kernel_size']
        activation = val['activation']
        inputs = x
        x = Conv1D(filters, kernel_size,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)    
    x = add([x, inputs])
    x = Activation(res_dict['last activation'])(x)    

    if nz_out == 1:
        '''use dense layer'''
        x = Dense(nChannel_out,activation=top_layer_dict['activation'])(x)
    else:
        '''use conv layer'''
        x = Conv1D(filters=nChannel_out,kernel_size=top_layer_dict['kernel_size'],padding='same',activation=top_layer_dict['activation'])(x)
    return Model(inputs=inputs,outputs=x)            
        

def _shortcut(input, residual):
    return Add()([input, residual])#, mode='sum')


def _bn_relu_conv(nb_filter, nb_row , strides=1, bn=False):
    def f(input):
        if bn:
            input = BatchNormalization()(input)
        activation = Activation('relu')(input)
        #activation=LeakyReLU(alpha=0.3)(input)
        return Conv1D( filters=nb_filter, kernel_size= nb_row, strides=strides, padding="same",use_bias=True)(activation)
    return f


def _residual_unit(nb_filter, filter_size, strides):
    def f(input):
        residual = _bn_relu_conv(nb_filter, filter_size )(input)
        residual = _bn_relu_conv(nb_filter, filter_size )(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, nb_filter, filter_size, repetations=1):
    def f(input):
        for i in range(repetations):
            strides = (1, 1)
            input = residual_unit(nb_filter=nb_filter, filter_size=filter_size,
                                  strides=strides)(input)
        return input
    return f


def ResCu(c_conf=(81,16,1,4), nb_residual_unit = 3, filter=128, filter_size=3):
    # conf = (nz_in,channel_in,nz_out,channel_out)
    
    # main input
    main_inputs = []
    outputs = []
    
    if c_conf is not None:
        nz_in,channel_in, nz_out,channel_out= c_conf
        input = Input(shape=(nz_in , channel_in))
        main_inputs=input
        
        
        conv1 = Conv1D(filters=filter,kernel_size=filter_size, padding="same")(input)
        residual_output = ResUnits(_residual_unit, nb_filter=filter, filter_size=filter_size,
                                   repetations=nb_residual_unit)(conv1)
        
        activation = Activation('relu')(residual_output)

        if nz_out == 1:
            # flat = tf.keras.layers.Flatten()(activation)
            outputs = Dense(channel_out,activation='tanh')(activation)
        else:     
            outputs = Conv1D(filters=channel_out, kernel_size=filter_size, padding="same",activation='tanh')(activation)
        

    
    model = Model(inputs=main_inputs, outputs=outputs)

    return model
