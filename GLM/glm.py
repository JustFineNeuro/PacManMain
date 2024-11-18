import pandas as pd
import arviz as az
import flax.linen as flax_nn
from jax import nn
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal, AutoMultivariateNormal, AutoLaplaceApproximation
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from optax import adam, chain, clip
import optax
import numpy as np
import patsy
import jax
import jax.numpy as jnp
from GLM.utils import PatsyTransformer, calculate_aic_bic_poisson
from GLM import models as mods
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score, GridSearchCV


# TODO: port compute tuningcurve for empirical comparison
# TODO: Out fo sample testing on models using X_test and y_test + setting distinct subsets of data without breaiking temporality
# TOdo: confidence bounds on predicitons might need to refit the model with statsmodels or bootstrap the standard error
# TODO circular shift
# TODO: stimulis history coding

class PoissonGLM:
    def __init__(self):
        '''

        :param spl_df: List indexing continuous variables and indexing number of spline bases to use
        :param spl_order: List indexing continuous variables and indexing order of spline bases to use
        '''
        self.fit_params = None
        self.test_size = None
        self.scores = None
        self.formulas = None
        self.pipeline = None
        self.X = None
        self.y = None

    def add_data(self, X=None, y=None):
        '''
        Add data before splitting
        :return:
        '''
        self.X = X
        self.y = y

        return self

    def split_test(self, test_size=0.2):
        '''
        Add data before splitting
        :return:
        '''
        self.test_size = test_size
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                    random_state=42)

        return self

    def make_preprocessor(self, formulas=None, metric='cv', l2reg=0.0001, solver='lbfgs'):
        '''

        :param solver:
        :param metric: 'cv','score'
        :param l2reg: l2 regularization
        :param formulas: model formulas in patsy format
        :return:
        '''
        self.formulas = formulas
        if type(formulas) is list:
            '''
            For model selection 
            '''
            if metric == 'cv':
                # Build the pipeline
                self.pipeline = Pipeline([
                    ('patsy', PatsyTransformer(formula=formulas[0])),  # Placeholder formula
                    ('model', PoissonRegressor(alpha=l2reg, solver=solver))
                ])
            elif metric == 'score':
                #Make a list of pipelines to iterate over
                pipelines = []
                for formula in formulas:
                    pipelines.append(Pipeline([
                        ('patsy', PatsyTransformer(formula=formula)),  # Placeholder formula
                        ('model', PoissonRegressor(alpha=l2reg, solver=solver))
                    ]))
                self.pipeline = pipelines

        elif type(formulas) is str:
            self.pipeline = Pipeline([
                ('patsy', PatsyTransformer(formula=formulas)),  # Placeholder formula
                ('model', PoissonRegressor(alpha=l2reg, solver=solver))
            ])

        return self

    def fit(self, params={'cv': 5, 'shuffleTime': True}):
        '''
        Main call to fit a model
        :param params:
        :return:
        '''
        self.fit_params = params

        # Optimizing over different models via cross-validation
        if type(self.formulas) is list:
            # Do cross-validation metrics
            if params['cv'] > 0:
                self.scores = pd.DataFrame(columns=['mean', 'std', 'model'])
                # Setup all formula to optimize over
                param_grid = {
                    'patsy__formula': self.formulas
                }

                # Are we using TimeSeriesShuffle?
                if params['shuffleTime'] is True:
                    cv = TimeSeriesSplit(n_splits=params['cv'])

                    # Grid search
                    grid_search = GridSearchCV(
                        estimator=self.pipeline,
                        param_grid=param_grid,
                        scoring='neg_mean_poisson_deviance',
                        cv=cv,
                        n_jobs=-1
                    )

                else:
                    # Grid search
                    grid_search = GridSearchCV(
                        estimator=self.pipeline,
                        param_grid=param_grid,
                        scoring='neg_mean_poisson_deviance',
                        cv=params['cv'],
                        n_jobs=-1
                    )

                grid_search.fit(self.X, self.y)
                cv_results = pd.DataFrame(grid_search.cv_results_)
                self.scores = cv_results[['param_patsy__formula', 'mean_test_score', 'std_test_score']]
                self.scores = self.scores.rename(columns={'param_patsy__formula': 'model'})
                self.best_fit_from_search = grid_search.best_estimator_.fit(self.X, self.y)
                self.best_pipeline = grid_search.best_estimator_

            elif params['cv'] <= 0:

                self.scores = pd.DataFrame(columns=['aic', 'bic', 'model'])

                for i, pipeline in enumerate(self.pipeline, 1):
                    #Fit the formula
                    pipeline.fit(self.X, self.y)

                    # Get pipeline info
                    model = pipeline.named_steps['model']  # Adjust based on the actual name in your pipeline
                    patsy_transformer = pipeline.named_steps['patsy']
                    k = patsy_transformer.transform(self.X).shape[1] + 1
                    #Make predicted data
                    y_pred = pipeline.predict(self.X)
                    [aic, bic] = calculate_aic_bic_poisson(len(self.y), mean_poisson_deviance(self.y, y_pred), k)
                    self.scores.loc[len(self.scores)] = [aic, bic, patsy_transformer.formula]
                    #TODO:  self.best_from_search

        elif type(self.formulas) is str:
            NotImplemented
            # TODO fit a single model
            #TODO: Cv (time or kfold) through cross-val score, or AIC/BIC
        #     if params['shuffleTime'] is True:
        #         cv = TimeSeriesSplit(n_splits=params['cv'])
        #
        # score = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_poisson_deviance').mean()
        # scores[f'Model {i}'] = score
        # self.pipeline.fit(self.X, self.y)
        return self

    def predict(self, pred_data, predict_params={'data': 'X', 'whichmodel': 'best'}):
        #TODO: make an intelligent search over all model terms and make predicted tuning curves
        '''
            predict at specific levels to make estimated curves to comapre against empirical
        '''
        design_matrix = self.best_pipeline[:-1].transform(pred_data)  # Transform without the final model step
        model = self.best_pipeline.named_steps['model']
        linear_predictor = model.intercept_ + design_matrix.dot(model.coef_)
        self.predicted_y = np.exp(linear_predictor)  # Poisson GLM applies exponential to linear predictor


class PoissonGLMbayes:

    def __init__(self):
        '''

        :param spl_df: List indexing continuous variables and indexing number of spline bases to use
        :param spl_order: List indexing continuous variables and indexing order of spline bases to use
        '''
        self.mcmc_result = None
        self.svi_result = None
        self.model = None
        self.fit_params = None
        self.test_size = None
        self.scores = None
        self.formulas = None
        self.pipeline = None
        self.X = None
        self.y = None

    def add_data(self, X=None, y=None):
        '''
        Add data before splitting
        :return:
        '''
        self.X = X
        self.y = y

        return self

    def split_test(self, test_size=0.2):
        '''
        Add data before splitting
        :return:
        '''
        self.test_size = test_size
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                    random_state=42)

        return self

    def define_model(self, model='gaussian_prior', basis_x_list=None, S_list=None, tensor_basis_list=None,
                     S_tensor_list=None,cat_basis=None):
        '''

        :param metric: 'cv','score'
        :param l2reg: l2 regularization
        :param formulas: model formulas in patsy format
        :return:
        '''
        self.basis_x_list = basis_x_list
        self.S_list = S_list
        self.tensor_basis_list = tensor_basis_list
        self.S_tensor_list = S_tensor_list
        self.cat_basis_list = cat_basis
        self.model = getattr(mods, model)

        return self

    def fit(self, PRNGkey=0, params={'fittype': 'mcmc', 'warmup': 500, 'mcmcsamples': 2000, 'chains': 1},baselinemodel=False, **kwargs):
        '''
        Main call to fit a model
        :param PRNGkey: range random key in jax format
        :param params:
        :param **kwargs accepted per model are:
            ardG_prs_mcmc: fit_intercept=True, lambda_param=0.1, sigma=1.0
            ardInd_prs_mcmc: fit_intercept=True, lambda_param=0.1, sigma=1.0
            prs_hyperlambda_mcmc: fit_intercept=True,cauchy=5.0,sigma=1.0
            gaussian_prior: fit_intercept=True,prior_scale=0.01
            laplace_prior: fit_intercept=True,prior_scale=0.01

        :return:
        '''

        if baselinemodel is True:
            self.noise_guide = AutoNormal(mods.baseline_noise_model)
            optimizer = optim.ClippedAdam(step_size=1e-2)

            svi = SVI(mods.baseline_noise_model, self.noise_guide, optimizer, loss=Trace_ELBO())
            self.noise_result = svi.run(jax.random.PRNGKey(0), 2000, y=jnp.array(self.y))

        elif baselinemodel is False or None:
            if str.lower(params['fittype']) == 'mcmc':
                nuts_kernel = NUTS(self.model)
                mcmc = MCMC(nuts_kernel, num_warmup=params['warmup'], num_samples=params['mcmcsamples'])
                mcmc.run(jax.random.PRNGKey(PRNGkey), basis_x_list=self.basis_x_list, S_list=self.S_list, y=self.y,
                         tensor_basis_list=self.tensor_basis_list, S_tensor_list=self.S_tensor_list, cat_basis=self.cat_basis_list, jitter=1e-6, **kwargs)

                self.mcmc_result = mcmc
            elif str.lower(params['fittype']) == 'vi':
                if params['guide'] == 'normal':
                    self.guide = AutoNormal(self.model)
                elif params['guide'] == 'mvn':
                    self.guide = AutoMultivariateNormal(self.model)
                elif params['guide'] == 'lap':
                    self.guide = AutoLaplaceApproximation(self.model)

                # Choose learning type and parameterization
                if 'lrate' not in params:
                    params['lrate'] = 0.01

                if 'optimtype' not in params:
                    optimizer = optim.ClippedAdam(step_size=params['lrate'])
                elif params['optimtype'] == 'scheduled':
                    # Define an exponential decay schedule for the learning rate
                    learning_rate_schedule = optax.exponential_decay(
                        init_value=1e-3,  # Starting learning rate
                        transition_steps=1000,  # Steps after which the rate decays
                        decay_rate=0.9,  # Decay factor for the learning rate
                        staircase=True  # If True, the decay happens in steps (discrete) rather than continuous
                    )
                    svi = SVI(self.model, self.guide, chain(clip(10.0), adam(learning_rate_schedule)), loss=Trace_ELBO())
                elif params['optimtype'] == 'fixed':
                    optimizer = optim.ClippedAdam(step_size=params['lrate'])
                    svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

                # Run inference
                self.svi_result = svi.run(jax.random.PRNGKey(0), params['visteps'], basis_x_list=self.basis_x_list,
                                          S_list=self.S_list, y=self.y,
                                          tensor_basis_list=self.tensor_basis_list, S_tensor_list=self.S_tensor_list,
                                          cat_basis=self.cat_basis_list, jitter=1e-6, **kwargs)
            self.fit_params = params

        return self

    def sample_posterior(self, nsamples=4000,baselinemodel=False):
        '''
            sample the posterior to compute relevant quantities
        :param nsamples:
        :return:
        '''

        if baselinemodel is not True:
            if self.fit_params['fittype'] == 'mcmc':
                self.posterior_samples = self.mcmc_result.posterior.get_samples()
            else:
                _posterior_samples = self.guide.sample_posterior(jax.random.PRNGKey(1), self.svi_result.params,
                                                                 sample_shape=(nsamples,))

                self.posterior_samples = {key: _posterior_samples[key] for key in _posterior_samples if
                                           key.startswith("beta_") or key.startswith("intercept") or key.startswith("cat_")}
                self.npostsamples=nsamples
        else:
            _posterior_samples = self.noise_guide.sample_posterior(jax.random.PRNGKey(1),
                                                                      self.noise_result.params,
                                                                      sample_shape=(nsamples,))

            self.posterior_noise_samples = {key: _posterior_samples[key] for key in _posterior_samples if
                                      key.startswith("beta_") or key.startswith("intercept") or key.startswith("cat_")}

        return self

    def summarize_posterior(self, credible_interval=90, format='long'):
        lower = 0 + int((100-credible_interval)/2)
        upper = 100 - int((100-credible_interval)/2)

        self.posterior_means = {}
        self.posterior_medians = {}
        self.posterior_sd = {}
        self.posterior_ci_lower = {}
        self.posterior_ci_upper = {}

        for keys in self.posterior_samples.keys():
            self.posterior_means[keys] = jnp.mean(self.posterior_samples[keys], axis=0)
            self.posterior_medians[keys] = jnp.median(self.posterior_samples[keys], axis=0)
            self.posterior_sd[keys] = jnp.std(self.posterior_samples[keys], axis=0)
            self.posterior_ci_lower[keys] = jnp.percentile(self.posterior_samples[keys], lower, axis=0)
            self.posterior_ci_upper[keys] = jnp.percentile(self.posterior_samples[keys], upper, axis=0)

        return self


    def coeff_relevance(self):
        '''
        Which coefficients to keep or zero out
        :return:
        '''

        self.coef_keep={}
        for keys in self.posterior_samples.keys():
            self.coef_keep[keys] = np.logical_xor(self.posterior_ci_lower[keys]>0, self.posterior_ci_upper[keys]<0).astype(int)

        return self


    def model_metrics(self, metric='WAIC',getbaselinemetric=True):
        '''
        metrics to include, mcfadden R2 or pseudo r2, waic, loo
        :param n_samples:
        :param metric:
        :return:
        '''

        if self.fit_params['fittype'] == 'mcmc':
            ''
        elif self.fit_params['fittype'] == 'vi':
            ''

            pointwise_log_likelihood = []
            basiskeys = [key for key in self.posterior_samples if key.startswith('beta_beta_')]
            tensorkeys = [key for key in self.posterior_samples if key.startswith('beta_tensor_')]
            interceptkeys = [key for key in self.posterior_samples if key.startswith('intercept')]

            for i in range(self.npostsamples):

                for ii, key in enumerate(basiskeys):
                    if ii ==0:
                        linear_pred = jnp.dot(self.posterior_samples[key][i] , self.basis_x_list[ii].transpose())
                    else:
                        linear_pred += jnp.dot(self.posterior_samples[key][i] , self.basis_x_list[ii].transpose())


                for ii, key in enumerate(tensorkeys):
                    linear_pred += jnp.dot(self.posterior_samples[key][i] , self.tensor_basis_list[ii].transpose())

                for ii, key in enumerate(interceptkeys):
                    linear_pred +=self.posterior_samples[key][i]

                # Calculate log-likelihood for each data point under a Poisson likelihood
                log_likelihood = dist.Poisson(rate=jnp.exp(linear_pred)).log_prob(self.y)
                pointwise_log_likelihood.append(log_likelihood)

            pointwise_log_likelihood = jnp.array(pointwise_log_likelihood)

            # # Convert pointwise log-likelihood to ArviZ's InferenceData format
            #Since it's vi expand dimension to emulate a chain
            pointwise_log_likelihood = pointwise_log_likelihood[None, :, :]
            idata1 = az.from_dict(log_likelihood={"log_likelihood": pointwise_log_likelihood})
            self.model_waic = az.waic(idata1, pointwise=True)


            # # Compute LOO
            # loo = az.loo(idata)
            # print("LOO:", loo)


        if getbaselinemetric is True:

            ylen=self.y.shape[0]
            pointwise_log_likelihood = []
            for i in range(self.npostsamples):
                linear_pred=jnp.exp(jnp.repeat(self.posterior_noise_samples['intercept'][i], ylen))
                log_likelihood = dist.Poisson(rate=linear_pred).log_prob(self.y)
                pointwise_log_likelihood.append(log_likelihood)

            pointwise_log_likelihood = jnp.array(pointwise_log_likelihood)
            pointwise_log_likelihood = pointwise_log_likelihood[None, :, :]
            idata2 = az.from_dict(log_likelihood={"log_likelihood": pointwise_log_likelihood})
            self.noise_waic = az.waic(idata2, pointwise=True)

        self.comparison = az.compare({'model1': idata1, 'baseline': idata2}, ic="waic")

        self.model_idata=idata1
        self.noise_idata=idata2

        return self


    def predict(self):
        NotImplemented

        '''
            predict at specific levels to make estimated curves to comapre against empirical
        '''
