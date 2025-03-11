path = '~/Downloads/metrics_lr_tempt/'
files = list.files(path)

repetition = model = n_iw_samples = lr = decay = nitems = nclass= mirt_dim =
  lcamethod = lca_acc=mse_itempars=mse_theta=var_itempars=var_theta=bias_itempars=bias_theta=runtime=c()
for (file in files){
  metrics = read.table(paste0(path, file))
  split = strsplit(file, '_')[[1]]
  repetition = c(repetition, split[2])
  model = c(model, split[3])
  n_iw_samples = c(n_iw_samples, split[4])
  lr = c(lr, split[5])
  decay = c(decay, split[6])
  nitems = c(nitems, split[7])
  if (split[3] == 'LCA'){
    nclass = c(nclass, split[8])
    lcamethod = c(lcamethod, split[9])
    mirt_dim = c(mirt_dim, NA)
  }
  else if (split[3] == 'GDINA'){
    nclass = c(nclass, split[8])
    lcamethod = c(lcamethod, NA)
    mirt_dim = c(mirt_dim, NA)
  }
  else if (split[3] == 'MIXIRT'){
    nclass = c(nclass, 2)
    mirt_dim = c(mirt_dim, split[8])
    lcamethod = c(lcamethod, NA)
  }
  lca_acc = c(lca_acc, metrics$V1[1])
  mse_itempars = c(mse_itempars, metrics$V1[2])
  mse_theta = c(mse_theta, metrics$V1[3])
  var_itempars = c(var_itempars, metrics$V1[4])
  var_theta = c(var_theta, metrics$V1[5])
  bias_itempars = c(bias_itempars, metrics$V1[6])
  bias_theta = c(bias_theta, metrics$V1[7])
  runtime = c(runtime, metrics$V1[8])

}

results = data.frame(repetition, model, n_iw_samples, lr, decay, nitems, nclass, mirt_dim,
                       lcamethod, lca_acc=as.numeric(lca_acc), mse_itempars=as.numeric(mse_itempars), 
                     mse_theta=as.numeric(mse_theta), var_itempars=as.numeric(var_itempars),
                     var_theta=as.numeric(var_theta), bias_itempars=as.numeric(bias_itempars), 
                     bias_theta=as.numeric(bias_theta),runtime=as.numeric(runtime))

library(dplyr)
library(ggplot2)

# Compute mean and standard deviation of MSE for each combination of lr and decay
agg_results <- results %>%
  filter(n_iw_samples>1) %>%
  group_by(model, lr, decay, n_iw_samples, nitems, nclass, mirt_dim, lcamethod) %>%
  summarise(
    mean_mse_item = mean(mse_itempars, na.rm = TRUE),
    sd_mse_item = sd(mse_itempars, na.rm = TRUE),
    mean_mse_theta = mean(mse_theta, na.rm = TRUE),
    sd_mse_theta = sd(mse_theta, na.rm = TRUE),
    mean_lca_acc = mean(lca_acc, na.rm=TRUE),
    .groups = "drop"
  )

# View summary
print(agg_results)

x=agg_results %>%
  filter(nitems==20, model=='LCA', nclass==10)
