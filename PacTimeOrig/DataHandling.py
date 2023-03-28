import numpy as np
import pandas as pd
from scipy.io import loadmat
import mat73
import os


def dataloader(direc=os.getcwd(),typesub='NHP',subject='H',session = 1, suffix='Pac_dACC.mat'):
    '''
    Loads the matlab style datafile from memory, assuming it is older than V7.3
    :param direc:
    :param fname:
    :return:
    '''
    fname= direc.removesuffix('notebooks')+'\\data\\' + typesub + '\\' + subject + '\\' + '_'.join([str(session), subject,suffix])
    data = loadmat(fname)
    return data


def ExperimentVarsRetrieve(datafile):
    expvars = {}
    expvars["numPrey"] = list()
    expvars["numNPCs"] = list()
    expvars["valNPCs"] = list()
    expvars["veloNPCs"] = list()
    expvars["typeNPCs"] = list()
    expvars["reward"] = list()
    expvars["punish"] = list()
    expvars["time_resolution"] = list()
    for trial in range(len(pd.DataFrame(datafile['vars']))):
        d = pd.DataFrame(datafile['vars'][trial, 0][0])

        expvars["numPrey"].append([d['numPrey'][0][0][0]])
        expvars["numNPCs"].append([d['numNPCs'][0][0][0]])
        expvars["valNPCs"].append(list(d['valNPCs'][0][0]))
        expvars["veloNPCs"].append(list(d['veloNPCs'][0][0]))
        expvars["typeNPCs"].append(list(d['typeNPCs'][0][0]))
        expvars["reward"].append(d["reward"][0][0])
        try:
            expvars["punish"].append(d["punish"][0][0])
        except:
            expvars["punish"].append(np.array([0]))

        expvars["time_resolution"].append(d['time_res'][0].mean())

    nprey=pd.DataFrame(expvars["numPrey"])
    nprey.rename(columns={0: 'numprey'},inplace=True)

    npcs = pd.DataFrame(expvars["numNPCs"])
    npcs.rename(columns={0: 'numNPC'}, inplace=True)

    val = pd.DataFrame(expvars["valNPCs"])
    val.rename(columns={0: 'NPCvalA', 1: 'NPCvalB'}, inplace=True)
    val['NPCvalB'] = val['NPCvalB'].fillna(0)  # On trials with a single target, replace the NaN with 0

    vel = pd.DataFrame(expvars["veloNPCs"])
    vel.rename(columns={0: 'NPCvelA', 1: 'NPCvelB'}, inplace=True)
    vel['NPCvelB'] = vel['NPCvelB'].fillna(0)  # On trials with a single target, replace the NaN with 0

    typeNPCs = pd.DataFrame(expvars["typeNPCs"])
    typeNPCs.rename(columns={0: 'NPCtypeA',1: 'NPCtypeB'}, inplace=True)
    typeNPCs['NPCtypeB'] = typeNPCs['NPCtypeB'].fillna(0) #On trials with a single target, replace the NaN with 0

    rwd = pd.DataFrame(expvars["reward"])
    rwd.rename(columns={0:'reward'},inplace=True)
    pun = pd.DataFrame(expvars["punish"])
    pun.rename(columns={0: 'punish'},inplace=True)
    session_vars=pd.concat([npcs,nprey,typeNPCs,val,vel,rwd,pun],axis=1)

    trialidx = pd.DataFrame(np.linspace(1, trial+1, trial+1))
    trialidx.rename(columns={0: 'trialidx'}, inplace=True)
    sessionidx=pd.DataFrame(np.ones(trial+1))
    sessionidx.rename(columns={0: 'sessionNumber'}, inplace=True)

    session_vars = pd.concat([session_vars,trialidx,sessionidx],axis=1)

    return session_vars


def retrievepositions(datafile):
    '''

    :param datafile:
    :return: returns the positions of joystick, prey and predator in a list, with each trial as a dataframe in the list containing all trials
    '''
    positions = list()

    for trial in range(len(pd.DataFrame(datafile['vars']))):
        d = pd.DataFrame(datafile['vars'][trial, 0][0])

        selfpos = pd.DataFrame(np.array(pd.DataFrame(d['self_pos'][0][0][0][:, :]).values)).rename(columns={0: 'selfXpos', 1: 'selfYpos'})

        # For prey/predator vars, let's make the number of columns equal for simplicity and index by expvars

        if d['numNPCs'][0][0][0] == 1:
            p1 = pd.DataFrame(d['prey_pos'][0][0][0]).rename(columns={0: 'prey1Xpos', 1: 'prey1Ypos'})
            p2 = pd.DataFrame(np.zeros((len(p1), 2)) * np.NAN).rename(columns={0: 'prey2Xpos', 1: 'prey2Ypos'})
            p3 = pd.DataFrame(np.zeros((len(p1), 2)) * np.NAN).rename(columns={0: 'predXpos', 1: 'predYpos'})
            pos = pd.concat([selfpos,p1, p2, p3], axis=1)
        elif d['numNPCs'][0][0][0] == 2 and d['numPrey'][0][0][0] == 2: # 2 prey scenario
            p1 = pd.DataFrame(d['prey_pos'][0][0][0]).rename(columns={0: 'prey1Xpos', 1: 'prey1Ypos'})
            p2 = pd.DataFrame(d['prey_pos'][0][0][1]).rename(columns={0: 'prey2Xpos', 1: 'prey2Ypos'})
            p3 = pd.DataFrame(np.zeros((len(p1),2))*np.NAN).rename(columns={0: 'predXpos', 1: 'predYpos'})
            pos = pd.concat([selfpos,p1, p2, p3], axis=1)
        elif d['numNPCs'][0][0][0] == 2 and d['numPrey'][0][0][0] == 1: #1 prey 1 pred scenario
            p1 = pd.DataFrame(d['prey_pos'][0][0][0]).rename(columns={0: 'prey1Xpos', 1: 'prey1Ypos'})
            p2 = pd.DataFrame(np.zeros((len(p1), 2)) * np.NAN).rename(columns={0: 'prey2Xpos', 1: 'prey2Ypos'})
            p3 = pd.DataFrame(d['pred_pos'][0][0][0]).rename(columns={0: 'predXpos', 1: 'predYpos'})
            pos = pd.concat([selfpos,p1, p2, p3], axis=1)

        positions.append(pos)

    return positions



