library(reticulate)
library(GDINA)
np <- import("numpy")

# read command line arguments
args = commandArgs(trailingOnly = TRUE)

# for now set arguments manually
NATTRIBUTES = args[1]
replication = args[2]
NITEMS = args[3]

expand_interactions <- function(attributes) {
  n_attributes <- ncol(attributes)
  n_effects <- 2^n_attributes - 1
  batch_size <- nrow(attributes)
  
  # Generate SxA matrix where each row represents whether each attribute is needed for each effect
  required_mask <- t(sapply(1:n_effects, function(x) as.integer(intToBits(x)[1:n_attributes])))
  
  # Repeat the matrix for each observation (BxSxA)
  required_mask <- array(rep(required_mask, each = batch_size), dim = c(batch_size, n_effects, n_attributes))
  
  # Repeat the observed attribute pattern for each possible combination (BxSxA)
  attributes <- aperm(abind::abind(replicate(n_effects, attributes, simplify=F),along = 3), c(1,3,2))
  
  # Set the observed attributes to 1 if they are not required for a pattern
  attributes[!required_mask] <- 1
  
  # Multiply over the different attributes, so that we get the probability of observing all necessary attributes
  effects <- apply(attributes, c(1, 2), prod)
  
  return(effects)
}

reverse_expand_interactions <- function(effects, n_attributes) {
  batch_size <- nrow(effects)
  n_effects <- 2^n_attributes - 1
  
  # Reconstruct the required_mask
  required_mask <- t(sapply(1:n_effects, function(x) as.integer(intToBits(x)[1:n_attributes])))
  
  # Initialize the Q matrix
  Q_matrix <- matrix(0, nrow = batch_size, ncol = n_attributes)
  
  # Iterate over each item (row in effects)
  for (i in 1:batch_size) {
    # Find which effects are active for this item
    active_effects <- which(effects[i, ] == 1)
    
    # Determine which attributes are required for these active effects
    required_attributes <- colSums(required_mask[active_effects, , drop = FALSE]) > 0
    
    # Update the Q matrix
    Q_matrix[i, ] <- as.integer(required_attributes)
  }
  return(Q_matrix)
}

data = np$load(path.expand(paste0(c('~/Documents/GitHub/Discrete_VAEs/saved_data/GDINA/data/', NATTRIBUTES, '_', NITEMS, '_', replication, '.npy'), collapse = ''))) 
true_att = np$load(path.expand(paste0(c('~/Documents/GitHub/Discrete_VAEs/saved_data/GDINA/class/', NATTRIBUTES, '_', NITEMS,'.npy'), collapse = '')))
true_itempars = np$load(path.expand(paste0(c('~/Documents/GitHub/Discrete_VAEs/saved_data/GDINA/itempars/', NATTRIBUTES, '_', NITEMS,'.npy'), collapse = '')))
true_itempars= matrix(true_itempars, nrow=nrow(true_itempars))
true_delta = true_itempars[,-1]
true_intercepts = true_itempars[,1]

# compute the Q matrix of nitems x n interactions and use it to compute the q matrix of nitems x n attributes
Q = true_delta != 0
Q = reverse_expand_interactions(Q, NATTRIBUTES)

t1 = Sys.time()
mod1 <- GDINA(dat = data, Q = Q, model = "GDINA", att.dist = 'independent')
runtime = runtime = as.numeric(Sys.time()-t1,units="secs")

delta_est = coef(mod1, what='delta', simplify=T)
delta_est_mat = expand_interactions(Q)
#delta_est_mat[,c(3,4)] = delta_est_mat[,c(4,3)]
for (item in 1:length(delta_est)){
  obs_atts = delta_est_mat[item,]>0 # which attributes are observed
  delta_est_mat[item, which(obs_atts)] = delta_est[[item]][2:(sum(obs_atts)+1)] # fill delta mat with observed slopes
}
intercepts_est = as.vector(sapply(delta_est, function(x) x["d0"]))


itempars_est = cbind(intercepts_est, delta_est_mat)

mse_delta = mean((true_itempars[true_itempars!=0] - itempars_est[true_itempars!=0])^2)
att_est = personparm(mod1)


acc = mean(att_est==true_att)
mse_itempars = mean((true_itempars[true_itempars!=0] - itempars_est[true_itempars!=0])^2)
mse_theta = NA
var_itempars = var(itempars_est[true_itempars!=0])
var_theta = NA
bias_itempars = mean(itempars_est[true_itempars!=0] - true_itempars[true_itempars!=0])
bias_theta = NA 



par = c()
value = c()
par_i = c()
par_j = c()

class_prob = personparm(mod1, what='mp')
itempars_est = array(t(itempars_est), dim = c(NITEMS,1,ncol(true_itempars)))
estimates = list(class_prob, itempars_est)
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
                     'nclass'=NATTRIBUTES,
                     'n_rep'=NREP,
                     'replication'=replication,
                     'parameter'=par,
                     'i'=par_i,
                     'j'=par_j,
                     'value'=value)



# write estimates to file

print(paste0(c('./results/estimates/mmlestimates_GDINA_', NCLASS, '_', N, '_', replication, '_', NITEMS, '_', NREP, '.csv'), collapse=''))
write.csv(results, paste0(c('./results/estimates/mmlestiamtes_GDINA_', NCLASS, '_', N, '_', replication, '_', NITEMS, '_', NREP, '.csv'), collapse=''))

# write metrics to file

metrics = c(as.character(acc), as.character(mse_theta), as.character(mse_theta), as.character(var_itempars), as.character(var_theta),
            as.character(bias_itempars), as.character(bias_theta), as.character(runtime))
fileConn<-file(paste0(c('./results/metrics/mmlmetrics_GDINA_', NCLASS, '_', N, '_', replication, '_', NITEMS, '_', NREP, '.txt'), collapse=''))
writeLines(metrics ,fileConn)
close(fileConn)

