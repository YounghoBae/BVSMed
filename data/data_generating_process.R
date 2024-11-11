## Load library
library(MASS)

## Load required data
load("data/covM_new.RData")
load("data/idx.RData")

## Setting random seed
set.seed(16)

## Data generating process
# n : # of observations, q : # of mediators, p : # of confounders
n = 466; q = 298; p = 3

# X : confounders matrix
X = mvrnorm(n, mu = rep(0,p), Sigma = diag(p))

# treat : vector of treatment variable
treat = rnorm(n, 0.5*X[,1]+0.2*X[,2]+0.7*X[,3], 1)

beta0_vec = rep(0.1,q)
B         = matrix(0.1, nrow = p, ncol = q)

# setting true tau vector based on final_idx that is selected in high correaltion mediators group based on covariance matrix
tau = rep(0, q)
tau[final_idx] = c(rep(-0.12,5),rep(-0.08,5),rep(-0.04,5),rep(0.04,5),rep(0.08,5),rep(0.12,5))

M = matrix(0,n,q)
for(i in 1:n){
  M[i,] = mvrnorm(1, beta0_vec + tau*treat[i] + (t(B) %*% X[i,]), 0.5*covM)
}

# setting true delta vector based on final_idx that is selected in high correaltion mediators group based on covariance matrix
delta    = rep(0, q)
delta[final_idx] = c(rep(c(0.5,1,1.5,0,0),6))

alpha0   = 2
alpha    = rep(2, p)
alpha_p  = 2

sigmaSq  = 0.5

y = rep(0,n)
for(i in 1:n){
  y[i] = rnorm(1, alpha0 + t(delta)%*%M[i,] + t(alpha)%*%X[i,] + alpha_p*treat[i], sqrt(sigmaSq))
}

save(treat, M, X, y, file = "data/data.RData")
