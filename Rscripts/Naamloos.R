library(poLCA)
library(clue)
library(reticulate)
np <- import("numpy")
# data reading

args = commandArgs(trailingOnly = TRUE)

NCLASS = 8
replication = 1
NITEMS = 20

data = np$load(paste0(c('~/Documents/GitHub/Discrete_VAEs/saved_data/LCA/data/', NCLASS, '_', NITEMS, '_', replication, '.npy'), collapse = '')) +1
true_class = read.csv(paste0(c('~/Documents/GitHub/Discrete_VAEs/saved_data/LCA/class/', NCLASS, '_', NITEMS, '_', replication,'.csv'), collapse = ''), header=F)
true_probs = as.matrix(read.csv(paste0(c('~/Documents/GitHub/Discrete_VAEs/saved_data/LCA/itempars/', NCLASS, '_', NITEMS, '_', replication,'.csv'), collapse = ''), header=F))

# fit model
f <- as.formula(paste("cbind(", paste(colnames(data), collapse = ","), ") ~ 1"))
t1 = Sys.time()
lca = poLCA(f, data, nclass = NCLASS, nrep = NREP)
runtime = as.numeric(Sys.time()-t1,units="secs")

# save estimated parameters
est_probs =  t(as.matrix(do.call(rbind, lapply(lca$probs, function(mat) mat[, 2]))))
est_class_probs = lca$posterior

#plot(as.vector(est_probs), as.vector(true_probs))

# assign classes to fix label switching
new_order = as.vector(clue::solve_LSAP(abs(cor(t(est_probs), t(true_probs))), maximum = T))
true_probs = true_probs[new_order, ]
true_class = true_class[, new_order]


est_class = apply(est_class_probs, 1, which.max)
true_class = apply(true_class, 1, which.max)

acc = mean(est_class == true_class)
mse = mean((true_probs - est_probs)^2)


par = c()
value = c()
par_i = c()
par_j = c()
estimates = list(est_class_probs, est_class)
par_names = c('conditional', 'class')
for (i in 1:2){
  est = as.matrix(estimates[[i]])
  for (r in 1:nrow(est)){
    for (c in 1:ncol(est)){
      par = c(par, par_names[i])
      par_i = c(par_i, r)
      par_j = c(par_j, c)
      value = c(value, est[r, c])
    }
  }
}

results = data.frame('model'='LCA',
                     'nclass'=NCLASS,
                     'n_rep'=NREP,
                     'replication'=replication,
                     'parameter'=par,
                     'i'=par_i,
                     'j'=par_j,
                     'value'=value)



# write estimates to file

print(paste0(c('~/Documents/GitHub/LCA_VAE/results/estimates/est_lca_', NCLASS, '_', N, '_', replication, '_', NITEMS, '_', NREP, '.csv'), collapse=''))
write.csv(results, paste0(c('~/Documents/GitHub/LCA_VAE/results/estimates/est_lca_', NCLASS, '_', N, '_', replication, '_', NITEMS, '_', NREP, '.csv'), collapse=''))

# write metrics to file

fileConn<-file(paste0(c('~/Documents/GitHub/LCA_VAE/results/metrics/lca_', NCLASS, '_', N, '_', replication, '_', NITEMS, '_', NREP, '.txt'), collapse=''))
writeLines(c(as.character(acc), as.character(mse), as.character(runtime)),fileConn)
close(fileConn)