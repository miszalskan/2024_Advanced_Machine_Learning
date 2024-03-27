library(mldr)

logit <- function(x) {
  return(log(x / (1 - x)))
}

logistic <- function(x) {
  return(1 / (1 + exp(-x)))
}

logit_prime <- function(x) {
  return(1 / (x * (1 - x)))
}

V <- function(x) {
  return( x * (1 - x))
}

WLS <- function(X, W, z) {
  XtWX <- t(X) %*% W %*% X
  XtWX_inv <- solve(XtWX)
  XtWz <- t(X) %*% W %*% z

  return(XtWX_inv %*% XtWz)
}

ll <- function(p, y) {
  ll <- y * log(p) + (1 - y) * log(1 - p)
  return(sum(ll))
}

IRLS <- function(X, y) {
  mu <- rep(.5, length(y))
  delta <- 1
  i <- 1

  # initialize log likelihood to 0
  LL <- 0

  while (delta > .00001) {
    Z <- logit(mu) + (y - mu) * logit_prime(mu)
    W <- diag(length(Z)) * as.vector((1 / (logit_prime(mu)^2 * V(mu))))

    beta <- WLS(X = X, W = W, z = Z)
    eta <- X %*% beta
    mu <- logistic(eta)

    LL_old <- LL
    LL <- ll(p = mu, y = y)
    delta <- abs(LL - LL_old)

    print(paste0("Iteration: ", i, " LL:", LL, " Beta: ", paste0(beta, collapse = ",")))
    i <- i + 1
  }
}
Z <- logit(mu) + (y - mu) * logit_prime(mu)
W <- diag(length(Z)) * as.vector((1 / (logit_prime(mu)^2 * V(mu))))

X <- model.matrix(neuralgia.glm)
y <- as.numeric(neuralgia$Pain == "Yes")


df <- IRLS(X = X, y = y)
df <- iris
df$Species.
library(emmeans)
neuralgia.glm <- glm(Pain ~ Treatment * Sex + Age,
                     family = binomial(link = "logit"),
                     data = neuralgia)
summary(neuralgia.glm)$coe

df <- read.arff('/Users/zuzannaglinka/AdvancedML/project1/data/dataset_37_diabetes.arff')


