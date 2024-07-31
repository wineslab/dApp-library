# Author: Daniel Uvaydov
# Use to run DL network on time series IQs
# Network takes as input 1024 IQs in frequency domain
# Network outputs label for each sub-band or each IQ (google semantic segmentation)

import numpy as np
import torch
import os
from sklearn.preprocessing import normalize
from .model import U_Net

def aggregate(y_pred, nparts, inp_dim, scale_fact, stride, nclasses):

    all_pred = []
    for band in y_pred:
        start_idxs = np.arange(0,inp_dim*scale_fact-inp_dim+1, stride)
        idx_mat = np.array([np.arange(k,k+inp_dim) for k in start_idxs], dtype=int)
        y_pred_final = np.zeros([nclasses,inp_dim*scale_fact])
        for i in range(inp_dim*scale_fact):
            aggregate_idx = np.where(i == idx_mat)
            y_pred_final[:,i] = np.average(band[aggregate_idx[0],:,aggregate_idx[1]], axis=0)

        all_pred.append(y_pred_final)

    return np.array(all_pred)


def convert_output_to_dict(output):
    #Dictionary mapping protocols to labels
    label_dict_str= {
        'empty'     : 0,
        'wifi'      : 1,
        'lte'       : 2,
        'zigbee'    : 3,
        'lora'      : 4,
        'ble'       : 5
    }
    
    protocol_dict = {}
    for i, labels in enumerate(output):
        protocol_labels = [list(label_dict_str.keys())[label] for label in labels]
        protocol_dict[f'protocol_{i+1}'] = protocol_labels
    return protocol_dict

def evaluate_iq_samples(id_gpu:str='0', samp_rate:int=25, input:str='./test_data.bin', 
                        model_path:str='/home/wineslab/spear-dApp/spear/models/best_model_dan.pth',
                        normalize:bool=False):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = id_gpu

    model = U_Net(2, 5, is_attention=True, alpha=1, beta=5)
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    #############Constants#############

    #Number of overt classes
    nclasses = 6
    #Bandwidth
    bw = samp_rate * 1e6
    #Scale factor from original 25MHz model was trained on
    scale_fact = int(bw//25e6)
    #Number of IQs network takes as input
    inp_dim = 1024
    #2 channels one for I and one for Q
    nchannels = 2
    #stride for sliding in mhz in BW is higher than 25MHz
    stride_mhz = 12.5
    stride = int((inp_dim*scale_fact)*(stride_mhz/samp_rate))
    #Folder containing IQ dat file
    iq_fp = input

    #Frequency array in MHz
    freq_arr = np.linspace(-(bw/1e6)/2, (bw/1e6)/2, int(scale_fact*inp_dim))

    #Open IQ file
    with open(iq_fp) as binfile:
        all_samps = np.fromfile(binfile, dtype=np.complex64, count=-1, offset=0)

    assert (len(all_samps)%inp_dim == 0), "Input needs to be multiple of 1024"

    #reshape to put in DL network
    all_samps = np.reshape(all_samps, [-1, inp_dim*scale_fact])

    #Perform fft and fftshift on each sample
    all_samps_frequency = np.fft.fft(all_samps)
    all_samps_frequency = np.fft.fftshift(all_samps_frequency, axes=1)

    if scale_fact > 1:
        all_samps_frequency_strided = []
        for samp in all_samps_frequency:
            samp = np.array([samp[k:k + inp_dim] for k in range(0, len(samp) - inp_dim + 1, stride)])
            all_samps_frequency_strided.append(samp)
        all_samps_frequency_strided = np.array(all_samps_frequency_strided)
        nparts = all_samps_frequency_strided.shape[1]

        all_samps_frequency_strided = np.stack((np.real(all_samps_frequency_strided), np.imag(all_samps_frequency_strided)), axis=-1)
        if normalize:
            all_samps_frequency_strided = normalize(np.reshape(all_samps_frequency_strided, (-1, 2)))
            all_samps_frequency_strided = np.reshape(all_samps_frequency_strided, (-1, 1024, 2))
        all_samps_frequency_strided = np.swapaxes(all_samps_frequency_strided, 1,2)
        all_samps_frequency_strided = torch.from_numpy(all_samps_frequency_strided)

        y_pred = model(all_samps_frequency_strided)
        y_pred = y_pred.detach().cpu().numpy()
        y_pred = np.reshape(y_pred, [-1,nparts,nclasses,inp_dim])
        y_pred = aggregate(y_pred, nparts, inp_dim, scale_fact, stride, nclasses)
        y_pred = np.swapaxes(y_pred, 1,2)

    else:
        all_samps_frequency = np.stack((np.real(all_samps_frequency), np.imag(all_samps_frequency)), axis=-1)
        if normalize:
            all_samps_frequency = normalize(np.reshape(all_samps_frequency, (-1, 2)))
            all_samps_frequency = np.reshape(all_samps_frequency, (-1, 1024, 2))
        all_samps_frequency = np.swapaxes(all_samps_frequency, 1,2)
        all_samps_frequency = torch.from_numpy(all_samps_frequency)

        y_pred = model(all_samps_frequency)
        y_pred = y_pred.detach().cpu().numpy()
        y_pred = np.swapaxes(y_pred, 1,2)

    #final output should be of size [batch_size, 1024]
    y_pred = np.argmax(y_pred, axis=-1)
    return y_pred, convert_output_to_dict(y_pred)
