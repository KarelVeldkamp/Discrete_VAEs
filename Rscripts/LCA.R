library(poLCA)
library(clue)
library(reticulate)
np <- import("numpy")
# data reading

args = commandArgs(trailingOnly = TRUE)

NCLASS = as.numeric(args[1])
replication = args[2]
NITEMS = as.numeric(args[3])
NREP=1

data = np$load(path.expand(paste0(c('./saved_data/LCA/data/', NCLASS, '_', NITEMS, '_', replication, '.npy'), collapse = ''))) +1
true_class = np$load(path.expand(paste0(c('./saved_data/LCA/class/', NCLASS, '_', NITEMS,'.npy'), collapse = '')))
true_probs = np$load(path.expand(paste0(c('./saved_data/LCA/itempars/', NCLASS, '_', NITEMS,'.npy'), collapse = '')))
true_probs= matrix(true_probs, nrow=nrow(true_probs))



# fit model
data = data.frame(data)
f <- as.formula(paste("cbind(", paste(colnames(data), collapse = ","), ") ~ 1"))
t1 = Sys.time()
lca = poLCA(f, data, nclass = NCLASS, nrep = NREP)
runtime = as.numeric(Sys.time()-t1,units="secs")

# save estimated parameters
est_probs =  t(as.matrix(do.call(rbind, lapply(lca$probs, function(mat) mat[, 2]))))
est_class_probs = lca$posterior

#plot(as.vector(est_probs), as.vector(true_probs))

# assign classes to fix label switching
new_order = as.vector(clue::solve_LSAP(abs(cor(t(est_probs), true_probs)), maximum = T))
true_probs = true_probs[, new_order]
true_class = true_class[, new_order]


est_class = apply(est_class_probs, 1, which.max)
true_class = apply(true_class, 1, which.max)

acc = mean(t(est_class) == true_class)
mse_itempars = mean((true_probs - t(est_probs))^2)
mse_theta = NA
var_itempars = var(est_probs)
var_theta = NA
bias_itempars = mean(t(est_probs) - true_probs)
bias_theta = NA 


par = c()
value = c()
par_i = c()
par_j = c()

est_probs = array(t(est_probs), dim = c(NITEMS,1,NCLASS))
estimates = list(est_class_probs, est_probs)
par_names = c('class', 'itempars')
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

print(paste0(c('./results/estimates/mmlmetrics_LCA_', NCLASS, '_', N, '_', replication, '_', NITEMS, '_', NREP, '.csv'), collapse=''))
write.csv(results, paste0(c('./results/estimates/mmlmetrics_LCA_', NCLASS, '_', N, '_', replication, '_', NITEMS, '_', NREP, '.csv'), collapse=''))

# write metrics to file

metrics = c(as.character(acc), as.character(mse_theta), as.character(mse_theta), as.character(var_itempars), as.character(var_theta),
            as.character(bias_itempars), as.character(bias_theta), as.character(runtime))
fileConn<-file(paste0(c('./results/metrics/mmlmetrics_LCA__', NCLASS, '_', N, '_', replication, '_', NITEMS, '_', NREP, '.txt'), collapse=''))
writeLines(metrics ,fileConn)
close(fileConn)