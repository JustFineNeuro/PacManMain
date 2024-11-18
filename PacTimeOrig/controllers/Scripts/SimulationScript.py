import time
import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm
from itertools import product
from PacTimeOrig.controllers import simulator as sim
from PacTimeOrig.controllers import JaxMod as jm
from PacTimeOrig.controllers import models as mods
from PacTimeOrig.controllers import utils as ut
from PacTimeOrig.data import scripts
import multiprocessing as mp



# TODO implement cross-test validation.

def simulate(cfgparams):
    results = pd.DataFrame(columns=['model','nrbf','opttype','gpscaler','runidx','gainmse','tlength','runtime',
                                  'posmse','poscorr','wtcorr','wtmse'])
    # Use monkey data
    Xdsgn, kinematics, sessvars, psth = scripts.monkey_run(cfgparams)

    # trials to use
    trialidx = np.sort(np.random.randint(1, len(Xdsgn), cfgparams['trials']))

    #Get system parameters
    A, B = ut.define_system_parameters()

    total_iterations = (
            len(cfgparams['opttype']) *
            len(cfgparams['models']) *
            len(cfgparams['rbfs']) *
            len(cfgparams['gpscaler']) *
            len(trialidx) * cfgparams['restarts']
    )

    with tqdm(total=total_iterations) as pbar:
        for optidx, opttype in enumerate(cfgparams['opttype']):
            for modidx, modname in enumerate(cfgparams['models']):
                for rbfidx, num_rbfs in enumerate(cfgparams['rbfs']):
                    for gpscalidx, gpscaler in enumerate(cfgparams['gpscaler']):
                        for _,trial in enumerate(trialidx):
                            for restart in range(cfgparams['restarts']):

                                # Get data
                                tdat = ut.trial_grab_kine(Xdsgn, trial)

                                #generate gains
                                L1,L2=ut.generate_sim_gains(cfgparams['ngains'][modidx])


                                if cfgparams['slack'] is False:
                                    #Simulate data
                                    if modname == 'p':
                                        outputs = sim.controller_sim_p(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
                                    elif modname == 'pv':
                                        outputs = sim.controller_sim_pv(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
                                    elif modname == 'pf':
                                        outputs = sim.controller_sim_pf(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
                                    elif modname == 'pvi':
                                        outputs = sim.controller_sim_pvi(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
                                    elif modname == 'pif':
                                        outputs = sim.controller_sim_pif(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
                                    elif modname == 'pvf':
                                        outputs = sim.controller_sim_pvf(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
                                    elif modname == 'pvif':
                                        outputs = sim.controller_sim_pvif(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)

                                # Make time
                                tmp = ut.make_timeline(outputs)

                                #Prep inputs
                                inputs = ut.prepare_inputs(A, B, outputs['x'], outputs['uout'], tdat['pry1_pos'], tdat['pry2_pos'], tmp, num_rbfs,
                                                           outputs['x'][:, 2:], tdat['pry1_vel'], tdat['pry2_vel'], pry_1_accel=tdat['pry1_accel'],
                                                           pry_2_accel=tdat['pry2_accel'])

                                #choose loss
                                if cfgparams['slack'] is False:
                                    loss_function = jm.create_loss_function_inner(ut.generate_rbf_basis, num_rbfs,
                                                                                  ut.generate_smoothing_penalty, lambda_reg=0.1,
                                                                                  ctrltype=modname, opttype=opttype)
                                elif cfgparams['slack'] is True:
                                    loss_function = jm.create_loss_function_inner_slack(ut.generate_rbf_basis, num_rbfs,
                                                                                  ut.generate_smoothing_penalty, lambda_reg=0.1,
                                                                                  ctrltype=modname, opttype=opttype)

                                #only used for trust
                                grad_loss = ut.compute_loss_gradient(loss_function)
                                hess_loss = ut.compute_hessian(loss_function)

                                if opttype == 'first':
                                    t1 = time.time()
                                    #######  use with ADAM   #######
                                    params = jm.initialize_parameters(inputs, ctrltype=modname, randomize_weights=True,
                                                                      slack_model=cfgparams['slack'])

                                    # Set up the optimizer
                                    optimizer, opt_state = jm.setup_optimizer(params, learning_rate=1e-2, slack_model= cfgparams['slack'],
                                                                              optimizer='adam')

                                    # Number of optimization steps
                                    num_steps = 10000

                                    # Optimization loop
                                    for step in range(num_steps):
                                        params, opt_state, best_loss = jm.optimization_step(params, opt_state, optimizer,
                                                                                       loss_function, inputs, ctrltype=modname,
                                                                                       slack_model=cfgparams['slack'])

                                        if step % 100 == 0:
                                            print(f"Step {step}, Loss: {best_loss}")


                                    runtime = time.time()-t1
                                elif opttype == 'second':
                                    t1 = time.time()
                                    #######  use with trust   #######
                                    params, best_params_flat, best_loss = jm.outer_optimization_lbfgs(inputs, loss_function, grad_loss,
                                                                                                  hess_loss,
                                                                                                  randomize_weights=True,
                                                                                              ctrltype=modname, maxiter=3000,
                                                                                              tolerance=1e-5, optimizer='trust',
                                                                                          slack_model=cfgparams['slack'])
                                    runtime = time.time()-t1

                                #Get parameters
                                if opttype == 'first':
                                    if cfgparams['slack'] is False:
                                        weights = params[0]
                                        width = params[1]
                                        # transform paramteres to correct domain
                                        L1_fit = np.array(jnp.log(1 + jnp.exp(params[2])))
                                        L2_fit = np.array(jnp.log(1 + jnp.exp(params[3])))
                                    elif cfgparams['slack'] is True:
                                        alpha = params[4]
                                elif opttype == 'second':
                                    if cfgparams['slack'] is False:
                                        weights = params[2]
                                        width = params[3]
                                        # transform paramteres to correct domain
                                        L1_fit = np.array(params[0])
                                        L2_fit = np.array(params[1])
                                    elif cfgparams['slack'] is True:
                                        alpha = params[4]

                                wtsim = ut.generate_sim_switch(inputs, width, weights)

                                if cfgparams['slack'] is False:
                                    shift = np.vstack((wtsim[0], wtsim[1]))
                                elif cfgparams['slack'] is True:
                                    shift = np.vstack((wtsim[0], wtsim[1], wtsim[2]))

                                # Sim for results test
                                if cfgparams['slack'] is False:
                                    # Simulate data
                                    if modname == 'p':
                                        output_pred = sim.controller_sim_p_post(tdat, shift, L1, L2, A=None, B=None)

                                    elif modname == 'pv':
                                        output_pred = sim.controller_sim_pv_post(tdat, shift, L1, L2, A=None, B=None)
                                    elif modname == 'pf':
                                        output_pred = sim.controller_sim_pf_post(tdat, shift, L1, L2, A=None, B=None)
                                    elif modname == 'pvi':
                                        output_pred = sim.controller_sim_pvi_post(tdat, shift, L1, L2, A=None, B=None)
                                    elif modname == 'pif':
                                        output_pred = sim.controller_sim_pif_post(tdat, shift, L1, L2, A=None, B=None)
                                    elif modname == 'pvf':
                                        output_pred = sim.controller_sim_pvf_psot(tdat, shift, L1, L2, A=None, B=None)
                                    elif modname == 'pvif':
                                        output_pred = sim.controller_sim_pvif_post(tdat, shift, L1, L2, A=None, B=None)

                                # compute metrics
                                gainmse = np.power(np.concatenate((L1-L1_fit,L2-L2_fit)),2).mean()
                                posmse = np.power(output_pred['x'][:,:2]-outputs['x'][:,:2],2).mean()
                                poscorr = np.corrcoef(output_pred['x'][:,:2].flatten(),outputs['x'][:,:2].flatten())[0,1]
                                wtmse = np.power(wtsim-outputs['shift'],2).mean()
                                wtcorr = np.corrcoef(np.array(wtsim).flatten(),outputs['shift'].flatten())[0,1]

                                new_row = {
                                    'model': modname,
                                    'nrbf': num_rbfs,
                                    'opttype': opttype,
                                    'gpscaler': gpscaler,
                                    'runidx': restart+1,
                                    'gainmse': gainmse,
                                    'tlength': outputs['x'].shape[0],
                                    'runtime': runtime,
                                    'posmse': posmse,
                                    'poscorr': poscorr,
                                    'wtcorr': wtcorr,
                                    'wtmse': wtmse
                                }
                                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
                                results.to_csv(
                                    '/Users/user/PycharmProjects/PacManMain/PacTimeOrig/controllers/results/maintest.csv')
                                pbar.update(1)


def process_simulation_task(args):
    """Process a single simulation task."""
    opttype, modname, num_rbfs, gpscaler, trial, restart, cfgparams, Xdsgn = args
    # Get data
    tdat = ut.trial_grab_kine(Xdsgn, trial)

    # generate gains
    L1, L2 = ut.generate_sim_gains(len(modname))

    if cfgparams['slack'] is False:
        # Simulate data
        if modname == 'p':
            outputs = sim.controller_sim_p(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
        elif modname == 'pv':
            outputs = sim.controller_sim_pv(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
        elif modname == 'pf':
            outputs = sim.controller_sim_pf(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
        elif modname == 'pvi':
            outputs = sim.controller_sim_pvi(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
        elif modname == 'pif':
            outputs = sim.controller_sim_pif(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
        elif modname == 'pvf':
            outputs = sim.controller_sim_pvf(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
        elif modname == 'pvif':
            outputs = sim.controller_sim_pvif(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)

    # Make time
    tmp = ut.make_timeline(outputs)
    A, B = ut.define_system_parameters()

    # Prep inputs
    inputs = ut.prepare_inputs(A, B, outputs['x'], outputs['uout'], tdat['pry1_pos'], tdat['pry2_pos'], tmp, num_rbfs,
                               outputs['x'][:, 2:], tdat['pry1_vel'], tdat['pry2_vel'], pry_1_accel=tdat['pry1_accel'],
                               pry_2_accel=tdat['pry2_accel'])

    # choose loss
    if cfgparams['slack'] is False:
        loss_function = jm.create_loss_function_inner(ut.generate_rbf_basis, num_rbfs,
                                                      ut.generate_smoothing_penalty, lambda_reg=0.1,
                                                      ctrltype=modname, opttype=opttype)
    elif cfgparams['slack'] is True:
        loss_function = jm.create_loss_function_inner_slack(ut.generate_rbf_basis, num_rbfs,
                                                            ut.generate_smoothing_penalty, lambda_reg=0.1,
                                                            ctrltype=modname, opttype=opttype)

    # only used for trust
    grad_loss = ut.compute_loss_gradient(loss_function)
    hess_loss = ut.compute_hessian(loss_function)

    if opttype == 'first':
        t1 = time.time()
        #######  use with ADAM   #######
        params = jm.initialize_parameters(inputs, ctrltype=modname, randomize_weights=True,
                                          slack_model=cfgparams['slack'])

        # Set up the optimizer
        optimizer, opt_state = jm.setup_optimizer(params, learning_rate=1e-2, slack_model=cfgparams['slack'],
                                                  optimizer='adam')

        # Number of optimization steps
        num_steps = 10000

        # Optimization loop
        for step in range(num_steps):
            params, opt_state, best_loss = jm.optimization_step(params, opt_state, optimizer,
                                                                loss_function, inputs, ctrltype=modname,
                                                                slack_model=cfgparams['slack'])

            if step % 100 == 0:
                print(f"Step {step}, Loss: {best_loss}")

        runtime = time.time() - t1
    elif opttype == 'second':
        t1 = time.time()
        #######  use with trust   #######
        params, best_params_flat, best_loss = jm.outer_optimization_lbfgs(inputs, loss_function, grad_loss,
                                                                          hess_loss,
                                                                          randomize_weights=True,
                                                                          ctrltype=modname, maxiter=3000,
                                                                          tolerance=1e-5, optimizer='trust',
                                                                          slack_model=cfgparams['slack'])
        runtime = time.time() - t1

    # Get parameters
    if opttype == 'first':
        if cfgparams['slack'] is False:
            weights = params[0]
            width = params[1]
            # transform paramteres to correct domain
            L1_fit = np.array(jnp.log(1 + jnp.exp(params[2])))
            L2_fit = np.array(jnp.log(1 + jnp.exp(params[3])))
        elif cfgparams['slack'] is True:
            alpha = params[4]
    elif opttype == 'second':
        if cfgparams['slack'] is False:
            weights = params[2]
            width = params[3]
            # transform paramteres to correct domain
            L1_fit = np.array(params[0])
            L2_fit = np.array(params[1])
        elif cfgparams['slack'] is True:
            alpha = params[4]

    wtsim = ut.generate_sim_switch(inputs, width, weights)

    if cfgparams['slack'] is False:
        shift = np.vstack((wtsim[0], wtsim[1]))
    elif cfgparams['slack'] is True:
        shift = np.vstack((wtsim[0], wtsim[1], wtsim[2]))

    # Sim for results test
    if cfgparams['slack'] is False:
        # Simulate data
        if modname == 'p':
            output_pred = sim.controller_sim_p_post(tdat, shift, L1, L2, A=None, B=None)

        elif modname == 'pv':
            output_pred = sim.controller_sim_pv_post(tdat, shift, L1, L2, A=None, B=None)
        elif modname == 'pf':
            output_pred = sim.controller_sim_pf_post(tdat, shift, L1, L2, A=None, B=None)
        elif modname == 'pvi':
            output_pred = sim.controller_sim_pvi_post(tdat, shift, L1, L2, A=None, B=None)
        elif modname == 'pif':
            output_pred = sim.controller_sim_pif_post(tdat, shift, L1, L2, A=None, B=None)
        elif modname == 'pvf':
            output_pred = sim.controller_sim_pvf_psot(tdat, shift, L1, L2, A=None, B=None)
        elif modname == 'pvif':
            output_pred = sim.controller_sim_pvif_post(tdat, shift, L1, L2, A=None, B=None)

    # compute metrics
    gainmse = np.power(np.concatenate((L1 - L1_fit, L2 - L2_fit)), 2).mean()
    posmse = np.power(output_pred['x'][:, :2] - outputs['x'][:, :2], 2).mean()
    poscorr = np.corrcoef(output_pred['x'][:, :2].flatten(), outputs['x'][:, :2].flatten())[0, 1]
    wtmse = np.power(wtsim - outputs['shift'], 2).mean()
    wtcorr = np.corrcoef(np.array(wtsim).flatten(), outputs['shift'].flatten())[0, 1]

    new_row = {
        'model': modname,
        'nrbf': num_rbfs,
        'opttype': opttype,
        'gpscaler': gpscaler,
        'runidx': restart + 1,
        'gainmse': gainmse,
        'tlength': outputs['x'].shape[0],
        'runtime': runtime,
        'posmse': posmse,
        'poscorr': poscorr,
        'wtcorr': wtcorr,
        'wtmse': wtmse
    }
    return new_row


def simulate_mp(cfgparams):
    """Simulate function with parallel processing."""
    Xdsgn, kinematics, sessvars, psth = scripts.monkey_run(cfgparams)
  # Placeholder for input data; replace with your actual data

    output_file = '/Users/user/PycharmProjects/PacManMain/PacTimeOrig/controllers/results/maintest.csv'  # Specify the output file path

    # Check if the results file exists; if not, create it with a header
    if not os.path.exists(output_file):
        pd.DataFrame(columns=[
            'model', 'nrbf', 'opttype', 'gpscaler', 'runidx',
            'gainmse', 'runtime', 'posmse', 'poscorr', 'wtcorr', 'wtmse'
        ]).to_csv(output_file, index=False)

    # Prepare tasks
    tasks = []
    for opttype in cfgparams['opttype']:
        for modname in cfgparams['models']:
            for num_rbfs in cfgparams['rbfs']:
                for gpscaler in cfgparams['gpscaler']:
                    for trial in range(cfgparams['trials']):
                        for restart in range(cfgparams['restarts']):
                            tasks.append((opttype, modname, num_rbfs, gpscaler, trial, restart, cfgparams, Xdsgn))

    # Parallel execution
    with mp.Pool(processes=mp.cpu_count()) as pool:
        with tqdm(total=len(tasks)) as pbar:
            for result in pool.imap_unordered(process_simulation_task, tasks):
                # Write each result immediately to the CSV file
                pd.DataFrame([result]).to_csv(output_file, mode='a', header=False, index=False)
                pbar.update(1)


    # results_list = []
    # with mp.Pool(processes=4) as pool:
    #     with tqdm(total=len(tasks)) as pbar:
    #         for result in pool.imap_unordered(process_simulation_task, tasks):
    #             results_list.append(result)
    #             pbar.update(1)

    # # Combine results into a DataFrame
    # results = pd.DataFrame(results_list)
    # results.to_csv('/Users/user/PycharmProjects/PacManMain/PacTimeOrig/controllers/results/maintest.csv', index=False)