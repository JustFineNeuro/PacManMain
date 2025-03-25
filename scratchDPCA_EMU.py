import numpy as np
import pandas as pd
from tqdm import tqdm
import dill as pickle
from dPCA import dPCA
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ChangeOfMind.functions import processing as proc

# Set parameters
cfgparams={}
cfgparams['event']='zero' # 'zero
cfgparams['keepamount']=10
cfgparams['timewarp']={}
cfgparams['prewin']=14
cfgparams['behavewin']=15
cfgparams['timewarp']['dowarp']=False
cfgparams['timewarp']['warpN']=cfgparams['prewin']+cfgparams['behavewin']+1
cfgparams['timewarp']['originalTimes']=np.arange(1,cfgparams['timewarp']['warpN']+1)
cfgparams['percent_train']=0.9
cfgparams['smoothing']=50
cfgparams['do_partial']=True


#Get data
datum='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/workspace.pkl'
ff=open(datum,'rb')
dat=pickle.load(ff)

#set metadata
metadata={}
#switch trajectories
switchtrajectory={}
for subj in dat['outputs_sess_emu'].keys():
    switchtrajectory[subj]={}


#Begin
metadata['good_session']={}
is_good=np.zeros((len(dat['vars_sess_emu'].keys()),1))
for i,subj in enumerate(dat['outputs_sess_emu'].keys()):
    if type(dat['vars_sess_emu'][subj][1]) is pd.DataFrame:
        metadata['good_session'][subj]=1

######

#Begin
#% make trial number check of trial types
#dataframe with monkey, session, type, and Number
metadata['trial_num']={}
tnum=pd.DataFrame(columns=['subject','session','switch_hilo_count','switch_lohi_count','total_neuron_count',
                           'neuron_count_acc','neuron_count_hpc','acc_index','hpc_index'])

for i,subj in enumerate(dat['outputs_sess_emu'].keys()):
    if metadata['good_session'][subj]==1:
        hilo=np.sum((dat['outputs_sess_emu'][subj][1]['splittypes'][:,1]==1).astype(int).reshape(-1,1) & (dat['outputs_sess_emu'][subj][1]['splittypes'][:,2]==1).astype(int).reshape(-1,1))
        lohi=np.sum((dat['outputs_sess_emu'][subj][1]['splittypes'][:,1]==1).astype(int).reshape(-1,1) & (dat['outputs_sess_emu'][subj][1]['splittypes'][:,2]==-1).astype(int).reshape(-1,1))
        areass = dat['brain_region_emu'][subj][1]
        hpc_count = np.where(np.char.find(areass, 'hpc') != -1)[0]
        acc_count = np.where(np.char.find(areass, 'acc') != -1)[0]

        new_row ={
                    'subject':subj,
                    'session':i,
                    'switch_hilo_count':hilo,
                    'switch_lohi_count':lohi,
                    'total_neuron_count':dat['psth_sess_emu'][subj][1][0].shape[1],
                    'neuron_count_acc': len(acc_count),
                    'neuron_count_hpc': len(hpc_count),
                    'acc_index':acc_count,
                    'hpc_index': hpc_count,

        }

        tnum=pd.concat([tnum,pd.DataFrame([new_row])],ignore_index=True)


#assign good sessions
tnum['use_sess']=(((tnum['switch_hilo_count'].values+tnum['switch_lohi_count'].values)/2)>cfgparams['keepamount']).astype(int).reshape(-1,1)

# assign
metadata['trial_num']=tnum

#Get all the neurons aligned and smoothed
neural_aligned={}
was_computed={}
for i,subj in enumerate(dat['outputs_sess_emu'].keys()):
    was_computed[subj]=[]
    if tnum.loc[tnum.subject==subj].use_sess.values[0] == 1:
        try:
            neural_aligned[subj]=[]
            psth=dat['psth_sess_emu'][subj]
            outputs_sess=dat['outputs_sess_emu'][subj]
            neural_aligned[subj] = proc.organize_neuron_by_split(psth,outputs_sess,cfgparams,[1],smoothwin=cfgparams['smoothing'])
            was_computed[subj] = 1
        except:
            was_computed[subj]=0


#
behavior_aligned = {}
was_computed = {}
for i, subj in enumerate(dat['outputs_sess_emu'].keys()):
    was_computed[subj] = []
    if tnum.loc[tnum.subject == subj].use_sess.values[0] == 1:
        try:
            behavior_aligned[subj] = []
            Xd = dat['Xd_sess_emu'][subj]
            outputs_sess = dat['outputs_sess_emu'][subj]
            behavior_aligned[subj] = proc.organize_behavior_by_split(Xd, outputs_sess, cfgparams, [1])

            was_computed[subj] = 1
        except:
            was_computed[subj] = 0


#
# # # # % time warp data
# #TODO: finihs behavior warp
if cfgparams['timewarp']['dowarp'] is True:
    for i,subj in enumerate(dat['outputs_sess_emu'].keys()):
        neural_aligned[subj]=proc.do_time_warp(neural_aligned[subj], cfgparams)




#Drop subjects with too low count
# subjkeep=metadata['trial_num']['subject'][np.where((metadata['trial_num']['switch_lohi_count']>18) &(metadata['trial_num']['switch_hilo_count']>18))[0]]
# neural_aligned = {key: neural_aligned[key] for key in subjkeep if key in neural_aligned}


inputs = neural_aligned

# % do mean dPCA
# rm=np.array([37,  59,  62,  63,  81, 123])
X_train_mean, _ = proc.decoding_prep(inputs, None, None, ismean=True, preptype='dpca')
X_train_mean = np.vstack(X_train_mean)
coefs=[]
if cfgparams['do_partial'] is True:
    X_speed_out = []
    intercepts_out=[]
    for _, subj in enumerate(inputs.keys()):
        X = behavior_aligned[subj][1]['speed'].squeeze()
        # X = 2*(np.abs((X>0.5).astype(int)-2)-1.5)
        for neuron in range(neural_aligned[subj][1]['fr'].shape[2]):
            X_speed_proj = np.zeros((30, 2))

            coefficients = np.zeros(30)  # One coefficient for each column
            intercepts = np.zeros(30)
            Y = neural_aligned[subj][1]['fr'][:, :, neuron].squeeze()

            for i in range(30):
                model = LinearRegression()
                model.fit(X[:, [i]], Y[:, i])  # Single feature X[:, i] predicts target Y[:, i]
                coefficients[i] = model.coef_[0]
                intercepts[i] = model.intercept_
            #Get speed projection
            a=X[np.where(behavior_aligned[subj][1]['direction'] == 1)[0], :].mean(axis=0) * coefficients+intercepts
            b=X[np.where(behavior_aligned[subj][1]['direction'] == -1)[0], :].mean(axis=0) * coefficients+intercepts
            X_speed_proj[:, 0] = a
            X_speed_proj[:, 1] = b
            X_speed_out.append(X_speed_proj)

    X_reldist_out = []

    for _, subj in enumerate(inputs.keys()):
        X = behavior_aligned[subj][1]['reldist'].squeeze()
        # X = 2*(np.abs((X>0.5).astype(int)-2)-1.5)
        for neuron in range(neural_aligned[subj][1]['fr'].shape[2]):
            X_reldist_proj = np.zeros((30, 2))

            coefficients = np.zeros(30)  # One coefficient for each column
            intercepts = np.zeros(30)

            Y = neural_aligned[subj][1]['fr'][:, :, neuron].squeeze()

            for i in range(30):
                model = LinearRegression()
                model.fit(X[:, [i]], Y[:, i])  # Single feature X[:, i] predicts target Y[:, i]
                coefficients[i] = model.coef_[0]
                intercepts[i] = model.intercept_


            # Get speed projection
            a = X[np.where(behavior_aligned[subj][1]['direction'] == 1)[0], :].mean(axis=0) * coefficients+intercepts
            b = X[np.where(behavior_aligned[subj][1]['direction'] == -1)[0], :].mean(axis=0) * coefficients+intercepts
            X_reldist_proj[:, 0] = a
            X_reldist_proj[:, 1] = b
            X_reldist_out.append(X_reldist_proj)

    partialer = np.stack(X_speed_out,axis=0).transpose((0,2,1))-np.stack(X_reldist_out,axis=0).transpose((0,2,1))


#Uncomment if all subjects
metadata['area_per_neuron']=flattened = np.concatenate([subdict[1] for subdict in dat['brain_region_emu'].values()])

# filtered_brain_region_emu = {key: dat['brain_region_emu'][key] for key in subjkeep if key in dat['brain_region_emu']}
# metadata['area_per_neuron'] = np.concatenate([subdict[1] for subdict in filtered_brain_region_emu.values()])


hpcidx=np.where(np.char.find(metadata['area_per_neuron'], 'hpc') != -1)[0]
accidx=np.where(np.char.find(metadata['area_per_neuron'], 'acc') != -1)[0]

#% do mean dPCA

dpca_params={}
dpca_params['mean_dPCA']=True
dpca_params['reg']=0.1
dpca_params['bias']=0.05
dpca_params['runs']=1
dpca_params['neur_idx']=hpcidx
dpca_params['inputs']=inputs
dpca_params['train_N']=None
dpca_params['test_N']=None
dpca_params['Vfull']=None
dpca_params['partialer']=partialer


Z_hpc,Vfull_hpc,expvar_hpc=proc.dpca_run(dpca_params)

dpca_params['neur_idx']=accidx

Z_acc,Vfull_acc,expvar_acc=proc.dpca_run(dpca_params)






#TODO: decompose neuron contributions


#### DOING DPCA here, I guess
#% dPCA with warping - convert to neurons x stim x time

filtered = metadata['trial_num'][
    (metadata['trial_num']['use_sess'] == 1) & (metadata['trial_num']['neuron_count_acc']>0)
]


# Do single trial DPCA on hpc
#COmpute number of training and test trials
dpca_params['train_N']=int(np.min([filtered.switch_hilo_count.min(),filtered.switch_lohi_count.min()])*cfgparams['percent_train'])
dpca_params['test_N']=int(np.min([filtered.switch_hilo_count.min(),filtered.switch_lohi_count.min()])*(1-cfgparams['percent_train']))+1

dpca_params['mean_dPCA']=False
dpca_params['reg']=1.0
dpca_params['bias']=0.05
dpca_params['runs']=250
dpca_params['neur_idx']=accidx
dpca_params['inputs']=inputs
dpca_params['Vfull']=Vfull_acc
dpca_params['partialer']=partialer

# DO full dpca test
# accuracy_acc = proc.dpca_run(dpca_params)

# DO SVM
decoding_params=dpca_params
decoding_params['do_pca']=True
decoding_params['ncomps']=15

decoding_acc=proc.decoding_run(decoding_params)

dpca_params['neur_idx']=hpcidx


decoding_hpc=proc.decoding_run(decoding_params)


