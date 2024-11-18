import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro
import matplotlib.pyplot as plt

from PacTimeOrig.data import DataHandling as dh
from PacTimeOrig.data import DataProcessing as dp
from PacTimeOrig.controllers import simulator as sim
from PacTimeOrig.controllers import JaxMod as jm
from PacTimeOrig.controllers import utils as ut



def monkey_run(cfgparams):
    # load datafile
    datafile = dh.dataloader(sess=cfgparams['session'], region=cfgparams['area'], subj=cfgparams['subj'],suffix='Pac_' + cfgparams['area'] + '.mat')

    # Get session variables
    sessvars = dh.getvars(datafile)

    # Get position data
    positions = dh.retrievepositions(datafile,dattype='nhp', rescale=[960.00, 540.00])
    if cfgparams['area'] == 'pMD':
        psth = dh.get_psth(datafile, win_shift=30)
    else:
        psth = dh.get_psth(datafile, win_shift=75)

    kinematics = dp.computederivatives(positions,
                                       vartodiff=['selfXpos', 'selfYpos', 'prey1Xpos', 'prey1Ypos', 'prey2Xpos',
                                                  'prey2Ypos'], dt=1.0 / 60.0)
    # sessvars = dp.get_reaction_time(sessvars,kinematics)
    sessvars = dh.get_matlab_wt_reaction_time(sessvars, session=cfgparams['session'], subj=cfgparams['subj'])
    # Select 2 prey trials
    ogsessvars = sessvars
    kinematics, sessvars = dh.subselect(kinematics, sessvars, trialtype='2')
    psth, _ = dh.subselect(psth, ogsessvars, trialtype='2')
    # Drop columns
    kinematics = dh.dropcols(kinematics, columns_to_drop=['predXpos', 'predYpos'])

    # Get W_t vector
    # wtvec = dh.get_wt_vector(folder_path='/Users/user/PycharmProjects/PacManMain/data/WtNHP/H/NHP_H_SESSION_3/',
    #                          selectype='average', transform=True)

    # Cut data to correct length of wt
    kinematics = dh.cut_to_rt(kinematics, sessvars)
    # psth = [pd.DataFrame(x) for x in psth]
    psth = dh.cut_to_rt(psth, sessvars)
    kinematics = dh.get_time_vector(kinematics)
    kinematics = dp.compute_distance(kinematics, trialtype=2)
    # compute relative normalized speed
    kinematics = dp.compute_relspeed(kinematics, trialtype=2)
    kinematics = dp.compute_selfspeed(kinematics)

    # For each kinematics frame, add relative reward value
    Xdsgn = kinematics
    for trial in range(len(Xdsgn)):
        Xdsgn[trial]['val1'] = np.repeat(sessvars.iloc[trial].NPCvalA, len(kinematics[trial]))
        Xdsgn[trial]['val2'] = np.repeat(sessvars.iloc[trial].NPCvalB, len(kinematics[trial]))

    # Switch reward positions so highest value is always in prey 1 slot
    Xdsgn = dh.rewardalign(Xdsgn)
    Xdsgn = [df[sorted(df.columns)] for df in Xdsgn]

    # Compute relative value
    Xdsgn = [df.assign(relvalue=df['val1'] - df['val2']).round(2) for df in Xdsgn]

    return Xdsgn, kinematics, sessvars, psth


def human_emu_run(cfgparams):


    # Dataloader, + #sessvar maker,
    sessvars, neural = dh.dataloader_EMU(folder=cfgparams['folder'], subj=cfgparams['subj'])

    dataall = neural['neuronData']

    # position getter and scaler (all alignedf already to chase_start)
    positions = dh.retrievepositions(dataall, dattype='hemu')

    # compute derivatives
    kinematics = dp.computederivatives(positions,
                                       vartodiff=['selfXpos', 'selfYpos', 'prey1Xpos', 'prey1Ypos', 'prey2Xpos',
                                                  'prey2Ypos'], dt=1.0 / 60.0, smooth=True)

    # get psth
    psth = dh.get_psth_EMU(dataall)

    # subselect 2 prey trials
    ogsessvars = sessvars
    kinematics, sessvars = dh.subselect(kinematics, sessvars, trialtype='2')
    psth, _ = dh.subselect(psth, ogsessvars, trialtype='2')

    # Rt compute and cut data
    sessvars = dp.get_reaction_time(sessvars, kinematics)
    kinematics = dh.cut_to_rt(kinematics, sessvars)
    psth = dh.cut_to_rt(psth, sessvars)
    kinematics = dh.get_time_vector(kinematics)
    kinematics = dp.compute_distance(kinematics, trialtype=2)
    # compute relative normalized speed
    kinematics = dp.compute_relspeed(kinematics, trialtype=2)
    kinematics = dp.compute_selfspeed(kinematics)

    # For each kinematics frame, add relative reward value
    Xdsgn = kinematics
    for trial in range(len(Xdsgn)):
        Xdsgn[trial]['val1'] = np.repeat(sessvars.iloc[trial].NPCvalA, len(kinematics[trial]))
        Xdsgn[trial]['val2'] = np.repeat(sessvars.iloc[trial].NPCvalB, len(kinematics[trial]))

    # Switch reward positions so highest value is always in prey 1 slot
    Xdsgn = dh.rewardalign(Xdsgn)
    Xdsgn = [df[sorted(df.columns)] for df in Xdsgn]

    # Compute relative value
    Xdsgn = [df.assign(relvalue=df['val1'] - df['val2']).round(2) for df in Xdsgn]

    return Xdsgn, kinematics, sessvars, psth


