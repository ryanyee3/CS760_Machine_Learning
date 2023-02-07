
M = matrix(c(5, 0, 0,
             0, 7, 0,
             0, 0, 3), nrow = 3, byrow= TRUE)

t(M) %*% M


library(MASS)

samples <- mvrnorm(100, mu = c(1, -1), Sigma = matrix(c(1, 0, 0, 1), nrow = 2))
plot(samples[,1], samples[,2])

unif_samples <- runif(100)
gm_samples <- matrix(rep(0, 200), nrow = 100)
for (i in 1:length(unif_samples)) {
  if (unif_samples[i] <= .3) {
    gm_samples[i, ] <- mvrnorm(1, mu = c(5, 0), Sigma = matrix(c(1, .25, .25, 1), nrow = 2))
  } else {
    gm_samples[i, ] <- mvrnorm(1, mu = c(-5, 0), Sigma = matrix(c(1, -.25, -.25, 1), nrow = 2))
  }
}

plot(gm_samples)
