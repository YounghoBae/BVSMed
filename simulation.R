library(MASS)
library(invgamma)
library(dplyr)
library(Rcpp)
sourceCpp("code_main/main.cpp")
load("data/data.RData")

# treat, M, X, y are generated data
# iter, burn_in, thin are iteration, burn_in, thining
# theta_gamma, theta_omega are hyper-parameter for tau, delta
# mu_lambda, h_lambda are hyper-parameter for lambda
# nu_element, psi_element are hpyer-parameter for tau, delta
# h0, c0, s0, t0, k0 are hpyer-parameter for beta0, B, alpha0, alpha, alpha_p
# V_tau, V_delta, V_lambda are standard deviation of proposal density for tau, delta, lambda
# mu0, mu1 are mean parameter for beta0, B
# nu0, sigmaSq0 are hyper-parameter for sigmaSq_Sigma that is variance of mediators model
# nu1, sigmaSq1 are hpyer-parameter for sigmaSq that is variance of outcome model
# update_prop can control whether update each parameter or not
# order of update_prop is (tau, delta, beta0, B, alpha0, alpha, alpha_p, lambda, sigmaSq_Sigma, sigmaSq)
# init is initial value for some paramters (sigmaSq, sigmaSq_Sigma, beta0, B, alpha0, alpha, alpha_p)
# eta is value that can be able to get from find_eta code

outcome = cmaVS_dep(treat=treat, M=M, X=X, y=y,
                    iter=300000, burn_in=150000, thin=150,
                    theta_gamma=-2.2, theta_omega=0.1,
                    mu_lambda=0, h_lambda=100,
                    nu_element=3, psi_element=3, 
                    h0=100, c0=100, s0=100, t0=100, k0=100,
                    V_tau=0.01, V_delta=0.01, V_lambda=0.01,
                    mu0=0, mu1=0, nu0=6, sigmaSq0=1/3, nu1=6, sigmaSq1=1/3,
                    update_prop=c(1,1,1,1,1,1,1,1,1,1), init = 0.3, eta=0.25)


