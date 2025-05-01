library(poLCA)
library(clue)
library(reticulate)
np <- import("numpy")
# data reading

# read command line arguments
args = commandArgs(trailingOnly = FALSE)

# set working directory to the directory above
filename = strsplit(args[grep("--file=", commandArgs(trailingOnly = FALSE))], '=')[[1]][2]
script_path <- normalizePath(filename)
script_dir <- dirname(script_path)
parent_dir <- dirname(script_dir)
setwd(parent_dir)

NCLASS = as.numeric(args[6])
replication = args[7]
NITEMS = as.numeric(args[8])
NREP=as.numeric(args[9])

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
var_itempars = var(as.vector(est_probs))
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

print(paste0(c('./results/estimates/mmlestimates_LCA_', NCLASS, '_', replication, '_', NITEMS, '_', NREP,'.csv'), collapse=''))
write.csv(results, paste0(c('./results/estimates/mmlestimates_LCA_', NCLASS, '_', replication, '_', NITEMS, '_', NREP, '.csv'), collapse=''))

# write metrics to file

metrics = c(as.character(acc), as.character(mse_itempars), as.character(mse_theta), as.character(var_itempars), as.character(var_theta),
            as.character(bias_itempars), as.character(bias_theta), as.character(runtime))
fileConn<-file(paste0(c('./results/metrics/mmlmetrics_LCA_', NCLASS, '_', replication, '_', NITEMS,'_', NREP, '.txt'), collapse=''))
writeLines(metrics ,fileConn)
close(fileConn)
