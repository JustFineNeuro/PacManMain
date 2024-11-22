import time
import os
import numpy as np
from tqdm import tqdm
from PacTimeOrig.controllers import JaxMod as jm
from PacTimeOrig.controllers import utils as ut
from PacTimeOrig.data import scripts
import argparse
import yaml
import pickle
import warnings

# Suppress all RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Load configuration from a YAML file
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_dict_loss(tmpresults):
    min_elbo_by_num_rbfs = {}
    # Iterate over unique num_rbfs values
    for nrbfs in set(key[0] for key in tmpresults.keys()):
        # Filter keys by the current num_rbfs and find the minimum elbo
        min_key = min(
            (key for key in tmpresults.keys() if key[0] == nrbfs),
            key=lambda k: tmpresults[k]['best_loss']
        )
        # Store the minimum elbo and its key
        min_elbo_by_num_rbfs[nrbfs] = {
            'key': min_key,
            'loss': tmpresults[min_key]['best_loss']
        }
    return min_elbo_by_num_rbfs


def run_fit(inputs, cfgparams, modname, modidx, num_rbfs):
    '''
    Runs that shit per iteration and should work with nhp and human
    :param inputs:
    :param cfgparams:
    :param modname:
    :param modidx:
    :param num_rbfs:
    :return:
    '''
    loss_function = jm.create_loss_function_inner_bayes(ut.generate_rbf_basis, num_rbfs,
                                                        ut.generate_smoothing_penalty,
                                                        lambda_reg=cfgparams['lambda_reg'],
                                                        ctrltype=modname, use_gmf_prior=True,
                                                        prior_std=cfgparams['prior_std'])

    # Compute jacobian and hessian
    grad_loss = ut.compute_loss_gradient(loss_function)
    hess_loss = ut.compute_hessian(loss_function)

    params, best_params_flat, best_loss = jm.outer_optimization_lbfgs(inputs, loss_function,
                                                                      grad_loss, hess_loss,
                                                                      randomize_weights=True,
                                                                      ctrltype=modname,
                                                                      maxiter=3000,
                                                                      tolerance=1e-5,
                                                                      optimizer='trust',
                                                                      slack_model=cfgparams[
                                                                          'slack'], bayes=True)

    prior_hessian = jm.compute_prior_hessian(prior_std=cfgparams['prior_std'], lambda_reg=cfgparams['lambda_reg'],
                                             num_weights=num_rbfs, num_gains=2 * cfgparams['ngains'][modidx],
                                             smoothing_matrix=ut.generate_smoothing_penalty(
                                                 num_rbfs))

    cov_matrix = jm.compute_posterior_covariance(hess_loss, best_params_flat, inputs,
                                                 prior_hessian)

    controller_trajectories = jm.simulate_posterior_samples(best_params_flat, cov_matrix,
                                                            inputs)

    # Compute the elbo
    elbo = jm.compute_elbo(cfgparams['prior_std'], best_params_flat, cov_matrix, inputs, modname,
                           num_samples=cfgparams['elbo_samples'])

    return params, best_params_flat,best_loss, prior_hessian,cov_matrix,controller_trajectories, elbo


def mainnhp(config_path):
    # Load config yaml file %OK
    config = load_config(config_path)
    cfgparams=config['cfgparams']
    # List subjects and hide hidden files %OK
    subjects = [f for f in os.listdir(config['data_path']) if not f.startswith('.')]
    print(subjects)
    #Loop over subjects
    for subjidx,subj in enumerate(subjects):
        cfgparams['subj']=subj
        sessfiles=[f for f in os.listdir(config['data_path'] + '/' + subj + '/' + config['cfgparams']['area'] + '/extended')
            if f.endswith('.mat')]

        nsessions=len(sessfiles)


        # routine:
        # 1. Fit a model 5x, store loss, elbo, trajectories
        # 2. Store the one with the best loss (store its elbo, trajectories and loss and cov matrix and parameters)
        # 3. Store best chosen model with BMS, do BMA over trajectories.
        results_save={}
        # Loop over session
        for sess in range(nsessions):
            # results_save['sess_'+str(sess)] = {}

            # +1 to deal witih file numbering
            cfgparams['session']=sess+1

            # get monkey data
            Xdsgn, kinematics, sessvars, psth = scripts.monkey_run(cfgparams)

            # Get system parameters
            A, B = ut.define_system_parameters()

            #Loop over trials
            for trial in tqdm(range(len(Xdsgn))):
                # results_save['sess_' + str(sess)]['trial_' + str(trial)]={}
                results_save['trial_' + str(trial)]={}

                results = {}

                # Loop over model
                for modidx, modname in enumerate(cfgparams['models']):

                    tmpresults={}
                    # loop over nrbfs
                    for num_rbfs in cfgparams['rbfs']:

                        #repeat process n times
                        for repeat in range(cfgparams['restarts']):

                            tdat = ut.get_data_for_fit(Xdsgn, trial)

                            tdat['x'] = np.hstack((tdat['player_pos'], tdat['player_vel']))

                            # Make time
                            tmp = ut.make_timeline(tdat)

                            # Prep inputs
                            inputs = ut.prepare_inputs(A, B, tdat['x'], tdat['uout'], tdat['pry1_pos'],
                                                       tdat['pry2_pos'], tmp, num_rbfs,
                                                       tdat['x'][:, 2:], tdat['pry1_vel'], tdat['pry2_vel'],
                                                       pry_1_accel=tdat['pry1_accel'],
                                                       pry_2_accel=tdat['pry2_accel'])

                            params, params_flat,best_loss, prior_hessian, cov_matrix, controller_trajectories, elbo\
                                =run_fit(inputs, cfgparams, modname, modidx, num_rbfs)

                            weights = params[2]
                            width = np.log(1+np.exp(params[3]))


                            wtsim = ut.generate_sim_switch(inputs,weights=weights,widths=width)
                            map_trajectory= np.vstack((wtsim[0], wtsim[1]))
                            # Store the results indexed by (num_rbfs, repeat)
                            tmpresults[(num_rbfs, repeat)] = {
                                'params': params,
                                'params_flat': params_flat,
                                'best_loss': best_loss,
                                'prior_hessian': prior_hessian,
                                'cov_matrix': cov_matrix,
                                'elbo': elbo,
                                'map_trajectory': map_trajectory,
                                'controller_trajectories': controller_trajectories
                            }

                        # choose best of run per nrbf
                        minlossbykey = get_dict_loss(tmpresults)
                        if len(cfgparams['rbfs'])==1:
                            results[modname] = tmpresults[minlossbykey[list(minlossbykey.keys())[0]]['key']]
                        else:
                            # choose 1 with highest elbo
                            elb=[]
                            hdimu=[]
                            for num_rbfs in cfgparams['rbfs']:
                                elb.append(tmpresults[minlossbykey[num_rbfs]['key']]['elbo'])
                                a=jm.compute_hdi(tmpresults[minlossbykey[num_rbfs]['key']]['controller_trajectories'][:, 0, :],
                                              0.95)[1]
                                b = jm.compute_hdi(tmpresults[minlossbykey[num_rbfs]['key']]['controller_trajectories'][:, 0, :],
                                                  0.95)[0]
                                hdimu.append(np.sum(a-b))

                            if cfgparams['uncertain_tiebreak'] is True:
                                selected = np.argmax(jm.compute_model_probabilities(elbos=[elb[0]-hdimu[0],elb[1]-hdimu[1]]))
                            else:
                                selected = np.argmax(elb)

                            results[modname]=tmpresults[minlossbykey[cfgparams['rbfs'][selected]]['key']]

                # MODEL SELECTION AND AVERAGING ##
                # save best in loss and best in elbo and BMA (re
                elb = []
                losses = []
                for key in results.keys():
                    elb.append(results[key]['elbo'])
                    losses.append(results[key]['best_loss'])

                to_save_best=results[cfgparams['models'][np.argmax(np.array(elb))]]
                to_save_best['name'] = cfgparams['models'][np.argmax(np.array(elb))]
                # results_save['sess_' + str(sess)]['trial_' + str(trial)]['best_elbo'] = to_save_best
                results_save['trial_' + str(trial)]['best_elbo'] = to_save_best

                to_save_best = results[cfgparams['models'][np.argmin(np.array(losses))]]
                to_save_best['name'] = cfgparams['models'][np.argmin(np.array(losses))]
                # results_save['sess_' + str(sess)]['trial_' + str(trial)]['best_loss'] = to_save_best
                results_save['trial_' + str(trial)]['best_loss'] = to_save_best

                #1. get model weights from elbos
                modprob = jm.compute_model_probabilities(np.array(elb))
                bma_traj=np.zeros_like(results[list(results.keys())[0]]['controller_trajectories'][:,0,:])
                for iter,key in enumerate(results.keys()):
                    bma_traj += modprob[iter]*results[key]['controller_trajectories'][:, 0, :]

                bma = np.stack((bma_traj, 1-bma_traj), axis=1)
                # results_save['sess_' + str(sess)]['trial_' + str(trial)]['bma']=bma

                results_save['trial_' + str(trial)]['bma']={}
                results_save['trial_' + str(trial)]['bma']['map'] = bma.mean(axis=0)
                results_save['trial_' + str(trial)]['bma']['hdi_high'] = jm.compute_hdi(bma[:,0,:])
                results_save['trial_' + str(trial)]['bma']['hdi_low'] = jm.compute_hdi(bma[:,1,:])

            #Pickle data per session
            fname=config['results_path']+subj+'_'+str(sess)+'_wt.pkl'

            with open(fname, 'wb') as f:
                pickle.dump(results_save, f, protocol=pickle.HIGHEST_PROTOCOL)

def mainnhp_persess(config_path):
    # Load config yaml file %OK
    config = load_config(config_path)
    cfgparams = config['cfgparams']
    subj = cfgparams['subj']
    sess = cfgparams['session']


    # get monkey data
    Xdsgn, kinematics, sessvars, psth = scripts.monkey_run(cfgparams)

    # Get system parameters
    A, B = ut.define_system_parameters()

    #Loop over trials
    # if 'check if finished?'
    #     else:
    trialsloop=len(Xdsgn)
    for trial in tqdm(range(trialsloop)):
        # results_save['sess_' + str(sess)]['trial_' + str(trial)]={}
        results_save={}

        results = {}

        # Loop over model
        for modidx, modname in enumerate(cfgparams['models']):

            tmpresults={}
            # loop over nrbfs
            for num_rbfs in cfgparams['rbfs']:

                #repeat process n times
                for repeat in range(cfgparams['restarts']):

                    tdat = ut.get_data_for_fit(Xdsgn, trial)

                    tdat['x'] = np.hstack((tdat['player_pos'], tdat['player_vel']))

                    # Make time
                    tmp = ut.make_timeline(tdat)

                    # Prep inputs
                    inputs = ut.prepare_inputs(A, B, tdat['x'], tdat['uout'], tdat['pry1_pos'],
                                               tdat['pry2_pos'], tmp, num_rbfs,
                                               tdat['x'][:, 2:], tdat['pry1_vel'], tdat['pry2_vel'],
                                               pry_1_accel=tdat['pry1_accel'],
                                               pry_2_accel=tdat['pry2_accel'])

                    params, params_flat,best_loss, prior_hessian, cov_matrix, controller_trajectories, elbo\
                        =run_fit(inputs, cfgparams, modname, modidx, num_rbfs)

                    weights = params[2]
                    width = np.log(1+np.exp(params[3]))


                    wtsim = ut.generate_sim_switch(inputs,weights=weights,widths=width)
                    map_trajectory= np.vstack((wtsim[0], wtsim[1]))
                    # Store the results indexed by (num_rbfs, repeat)
                    tmpresults[(num_rbfs, repeat)] = {
                        'params': params,
                        'params_flat': params_flat,
                        'best_loss': best_loss,
                        'prior_hessian': prior_hessian,
                        'cov_matrix': cov_matrix,
                        'elbo': elbo,
                        'map_trajectory': map_trajectory,
                        'controller_trajectories': controller_trajectories
                    }

                # choose best of run per nrbf
                minlossbykey = get_dict_loss(tmpresults)
                if len(cfgparams['rbfs'])==1:
                    results[modname] = tmpresults[minlossbykey[list(minlossbykey.keys())[0]]['key']]
                else:
                    # choose 1 with highest elbo
                    elb=[]
                    hdimu=[]
                    for num_rbfs in cfgparams['rbfs']:
                        elb.append(tmpresults[minlossbykey[num_rbfs]['key']]['elbo'])
                        a=jm.compute_hdi(tmpresults[minlossbykey[num_rbfs]['key']]['controller_trajectories'][:, 0, :],
                                      0.95)[1]
                        b = jm.compute_hdi(tmpresults[minlossbykey[num_rbfs]['key']]['controller_trajectories'][:, 0, :],
                                          0.95)[0]
                        hdimu.append(np.sum(a-b))

                    if cfgparams['uncertain_tiebreak'] is True:
                        selected = np.argmax(jm.compute_model_probabilities(elbos=[elb[0]-hdimu[0],elb[1]-hdimu[1]]))
                    else:
                        selected = np.argmax(elb)

                    results[modname]=tmpresults[minlossbykey[cfgparams['rbfs'][selected]]['key']]

        # MODEL SELECTION AND AVERAGING ##
        # save best in loss and best in elbo and BMA (re
        elb = []
        losses = []
        for key in results.keys():
            elb.append(results[key]['elbo'])
            losses.append(results[key]['best_loss'])

        to_save_best=results[cfgparams['models'][np.argmax(np.array(elb))]]
        to_save_best['name'] = cfgparams['models'][np.argmax(np.array(elb))]
        # results_save['trial_' + str(trial)]['best_elbo'] = to_save_best
        results_save['best_elbo'] = to_save_best

        to_save_best = results[cfgparams['models'][np.argmin(np.array(losses))]]
        to_save_best['name'] = cfgparams['models'][np.argmin(np.array(losses))]
        # results_save['trial_' + str(trial)]['best_loss'] = to_save_best
        results_save['best_loss'] = to_save_best

        #1. get model weights from elbos
        modprob = jm.compute_model_probabilities(np.array(elb))
        bma_traj=np.zeros_like(results[list(results.keys())[0]]['controller_trajectories'][:,0,:])
        for iter,key in enumerate(results.keys()):
            bma_traj += modprob[iter]*results[key]['controller_trajectories'][:, 0, :]

        bma = np.stack((bma_traj, 1-bma_traj), axis=1)
        # results_save['sess_' + str(sess)]['trial_' + str(trial)]['bma']=bma

        # results_save['trial_' + str(trial)]['bma']={}
        # results_save['trial_' + str(trial)]['bma']['map'] = bma.mean(axis=0)
        # results_save['trial_' + str(trial)]['bma']['hdi_high'] = jm.compute_hdi(bma[:,0,:])
        # results_save['trial_' + str(trial)]['bma']['hdi_low'] = jm.compute_hdi(bma[:,1,:])
        results_save['bma'] = {}
        results_save['map'] = bma.mean(axis=0)
        results_save['bma']['hdi_high'] = jm.compute_hdi(bma[:, 0, :])
        results_save['bma']['hdi_low'] = jm.compute_hdi(bma[:, 1, :])
        #Pickle data per session
        fname=config['results_path']+subj+'_'+str(sess)+'_'+str(trial+1)+'_wt.pkl'

        with open(fname, 'wb') as f:
            pickle.dump(results_save, f, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a model fit per session.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["main1", "main2"],
        help="Choose which main function to run: 'main1' or 'main2'."
    )
    args = parser.parse_args()

    # Call the appropriate main function based on --mode
    if args.mode == "main1":
        mainnhp(args.config)
    elif args.mode == "main2":
        mainnhp_persess(args.config)
    elif args.mode == "main3":
        NotImplemented
        # mainemu(args.config)




# def mainemu(config_path):
#     #Load config yaml file
#     config = load_config(config_path)
#
#     #List subjects and hide hidden files %OK
#     subjects = [f for f in os.listdir(config['data_path']) if not f.startswith('.')]
#
#     #humans only have 1 session
#     Xdsgn, kinematics, sessvars, psth = scripts.human_emu_run(cfgparams)