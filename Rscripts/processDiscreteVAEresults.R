path = '~/Downloads/metrics/'
files = list.files(path)

repetition = model = n_iw_samples = lr = decay = nitems = nclass= mirt_dim = est = 
  lcamethod = lca_acc=mse_itempars=mse_theta=var_itempars=var_theta=bias_itempars=bias_theta=runtime=c()
for (file in files){
  metrics = read.table(paste0(path, file))
  split = strsplit(file, '_')[[1]]
  est = c(est, ifelse(split[1]=='mmlmetrics', 'MML', 'VAE'))
  if (split[1] == 'metrics'){
    repetition = c(repetition, split[2])
    model = c(model, split[3])
    n_iw_samples = c(n_iw_samples, split[4])
    lr = c(lr, split[5])
    decay = c(decay, split[6])
    nitems = c(nitems, split[7])
    if (split[3] == 'LCA'){
      nclass = c(nclass, split[8])
      lcamethod = c(lcamethod, substr(split[9], 1, nchar(split[9])-4))
      mirt_dim = c(mirt_dim, NA)
    }
    else if (split[3] == 'GDINA'){
      nclass = c(nclass, split[8])
      lcamethod = c(lcamethod, '')
      mirt_dim = c(mirt_dim, '')
    }
    else if (split[3] == 'MIXIRT'){
      nclass = c(nclass, 2)
      mirt_dim = c(mirt_dim, split[8])
      lcamethod = c(lcamethod, '')
    }
  }
  else if(split[1]=='mmlmetrics'){
    model = c(model, split[2])
    
    n_iw_samples = c(n_iw_samples, '')
    decay = c(decay, '')
    lcamethod = c(lcamethod, '')
    lr = c(lr, '')
    nclass = c(nclass, split[3])
    repetition = c(repetition, split[4])
    nitems = c(nitems, substr(split[5], 1, nchar(split[5])-4))
    mirt_dim = c(mirt_dim, NA)
    
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

library(dplyr)
library(ggplot2)

results = data.frame(repetition, model, n_iw_samples, lr, decay, nitems, nclass, mirt_dim,
                       lcamethod, est, lca_acc=as.numeric(lca_acc), mse_itempars=as.numeric(mse_itempars), 
                     mse_theta=as.numeric(mse_theta), var_itempars=as.numeric(var_itempars),
                     var_theta=as.numeric(var_theta), bias_itempars=as.numeric(bias_itempars), 
                     bias_theta=as.numeric(bias_theta),runtime=as.numeric(runtime)) %>%
  filter(!(lcamethod=='vq' & n_iw_samples>1))





# # Compute mean and standard deviation of MSE for each combination of lr and decay
# agg_results <- results %>%
#   filter(lr==0.01) %>%
#   group_by(model, lr, decay, n_iw_samples, nitems, nclass, mirt_dim, lcamethod) %>%
#   summarise(
#     mean_mse_item = mean(mse_itempars, na.rm = TRUE),
#     sd_mse_item = sd(mse_itempars, na.rm = TRUE),
#     mean_mse_theta = mean(mse_theta, na.rm = TRUE),
#     sd_mse_theta = sd(mse_theta, na.rm = TRUE),
#     mean_lca_acc = mean(lca_acc, na.rm=TRUE),
#     .groups = "drop"
#   )

########################## LCA ###################################
# itempars plot LCA
results %>% 
  filter((decay == .9 &  lr==.01)| est == 'MML', model=='LCA') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw ', lcamethod),
         method = factor(method, levels = c("MML iw ", "VAE 10iw gs", "VAE 5iw gs", "VAE 1iw gs", "VAE 1iw vq"))
         ) %>%
  ggplot(aes(y=mse_itempars, x=method, col=method)) +
  facet_grid(nclass~nitems, scales = 'free_y', ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() +
  ylim(0,0.001)

# acc plot LCA
results %>% 
  filter((decay == .9 &  lr==.01)| est == 'MML', model=='LCA') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw ', lcamethod),
         method = factor(method, levels = c("MML iw ", "VAE 10iw gs", "VAE 5iw gs", "VAE 1iw gs", "VAE 1iw vq"))
  ) %>%
  ggplot(aes(y=lca_acc, x=method, col=method)) +
  facet_grid(nclass~nitems, scales = 'free_y', ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() +
  ylim(0.9,1)

# runtime plot LCA
results %>% 
  filter((decay == .9 &  lr==.01)| est == 'MML', model=='LCA') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw ', lcamethod),
         method = factor(method, levels = c("MML iw ", "VAE 10iw gs", "VAE 5iw gs", "VAE 1iw gs", "VAE 1iw vq"))
  ) %>%
  ggplot(aes(y=runtime, x=method, col=method)) +
  facet_grid(nclass~nitems, scales = 'free_y', ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() 

########################## GDINA ###################################
# itempars plot GDINA
results %>% 
  filter((decay == .999 &  lr==.01)| est == 'MML', model=='GDINA') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw'),
         method = factor(method, levels = c("MML iw", "VAE 10iw", "VAE 5iw", "VAE 1iw"))
  ) %>%
  filter(mse_itempars<0.03) %>%
  ggplot(aes(y=mse_itempars, x=method, col=method)) +
  facet_grid(nclass~nitems, scales = 'free_y', ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() 

# acc plot GDINA
results %>% 
  filter((decay == .999 &  lr==.01)| est == 'MML', model=='GDINA') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw'),
         method = factor(method, levels = c("MML iw", "VAE 10iw", "VAE 5iw", "VAE 1iw"))
  ) %>%
  ggplot(aes(y=lca_acc, x=method, col=method)) +
  facet_grid(nclass~nitems, scales = 'free_y', ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() 

# runtime plot GDINA
results %>% 
  filter((decay == .999 &  lr==.01)| est == 'MML', model=='GDINA') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw'),
         method = factor(method, levels = c("MML iw", "VAE 10iw", "VAE 5iw", "VAE 1iw"))
  ) %>%
  ggplot(aes(y=runtime, x=method, col=method)) +
  facet_grid(nclass~nitems, scales = 'free_y', ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() 



######################### MIXIRT ############################
results %>% 
  filter((lr==.01), model=='MIXIRT') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw'),
         method = factor(method, levels = c("VAE 10iw", "VAE 5iw", "VAE 1iw"))
  ) %>%
  ggplot(aes(y=mse_itempars, x=method, col=method)) +
  facet_grid(mirt_dim~decay, scales = 'free_y', ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() 

  
