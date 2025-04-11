path = '~/Documents/discrete_simulation_results/'
files = list.files(path)

repetition = model = n_iw_samples = lr = decay = min_temp = nitems = nclass= mirt_dim = est = 
  lcamethod = lca_acc=mse_itempars=mse_theta=var_itempars=var_theta=bias_itempars=bias_theta=runtime= 
  mse_intercepts=c()
for (file in files){
  metrics = read.table(paste0(path, file))
  split = strsplit(file, '_')[[1]]
  est = c(est, ifelse(split[1]=='metrics', 'VAE', 'MML'))
  if (split[1] == 'metrics'){
    repetition = c(repetition, split[2])
    model = c(model, split[3])
    n_iw_samples = c(n_iw_samples, split[4])
    lr = c(lr, split[5])
    decay = c(decay, split[6])
    min_temp = c(min_temp, split[7])
    nitems = c(nitems, split[8])
    if (split[3] == 'LCA'){
      nclass = c(nclass, split[9])
      lcamethod = c(lcamethod, substr(split[10], 1, nchar(split[10])-4))
      mirt_dim = c(mirt_dim, NA)
      mse_intercepts = c(mse_intercepts, NA)
    }else if (split[3] == 'GDINA'){
      nclass = c(nclass, split[9])
      lcamethod = c(lcamethod, '')
      mirt_dim = c(mirt_dim, '')
      mse_intercepts = c(mse_intercepts, NA)
    }else if (split[3] == 'MIXIRT'){
      nclass = c(nclass, 2)
      mirt_dim = c(mirt_dim, split[9])
      lcamethod = c(lcamethod, '')
      mse_intercepts = c(mse_intercepts, metrics$V1[9])
    }
  }else if(split[1]=='mmlmetrics'){
    model = c(model, split[2])
    
    n_iw_samples = c(n_iw_samples, '')
    decay = c(decay, '')
    min_temp = c(min_temp, '')
    lcamethod = c(lcamethod, '')
    lr = c(lr, '')
    nclass = c(nclass, split[3])
    repetition = c(repetition, split[4])
    nitems = c(nitems, substr(split[5], 1, nchar(split[5])-4))
    mirt_dim = c(mirt_dim, NA)
    mse_intercepts = c(mse_intercepts, NA)
  }else if(split[1]=='mplus'){
    model = c(model, 'MIXIRT')
    n_iw_samples = c(n_iw_samples, '')
    decay = c(decay, '')
    min_temp = c(min_temp, '')
    lcamethod = c(lcamethod, '')
    lr = c(lr, '')
    nclass = c(nclass, 2)
    repetition = c(repetition, substr(split[3], 1, nchar(split[3])-4))
    ndim = split[2]
    nitems = c(nitems, ifelse(ndim==3, 28, 110))
    mirt_dim = c(mirt_dim, ndim)
    
    mse_intercepts = c(mse_intercepts, metrics$V1[17])
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

results = data.frame(repetition, model, n_iw_samples, lr, decay, min_temp, nitems, nclass, mirt_dim,
                       lcamethod, est, lca_acc=as.numeric(lca_acc), mse_itempars=as.numeric(mse_itempars), 
                     mse_theta=as.numeric(mse_theta), var_itempars=as.numeric(var_itempars),
                     var_theta=as.numeric(var_theta), bias_itempars=as.numeric(bias_itempars), 
                     bias_theta=as.numeric(bias_theta),runtime=as.numeric(runtime), mse_intercepts=as.numeric(mse_intercepts))





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
  filter((decay == .9 &  min_temp==.01&lr==.01 &lcamethod!='vq')| est == 'MML', model=='LCA') %>%
  mutate(method = paste0(est,' ', n_iw_samples,'iw ', lcamethod),
         method = factor(method, levels = c("MML iw ", "VAE 10iw gs", "VAE 5iw gs", "VAE 1iw gs"))) %>%
  ggplot(aes(y=mse_itempars, x=method, col=method)) +
  facet_grid(nclass~nitems, scales = "free_y") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text = element_text(size = 20)) +
  geom_boxplot() 


results %>%
  filter(decay == .9 &  lr==.01, lcamethod=='vq') %>%
  ggplot(aes(x=mse_itempars)) +
    geom_histogram() +
    facet_grid(nclass~nitems, scales = "free_y")  
hist(x$mse_itempars)


x = results %>%
  filter(model=='LCA', lcamethod=='gs', nitems==100, nclass==10, n_iw_samples==1, lr==0.01) 
plot(density(x$mse_itempars))

# acc plot LCA
results %>% 
  filter((decay == .9 &  lr==.01&lcamethod!='vq')| est == 'MML', model=='LCA') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw ', lcamethod),
         method = factor(method, levels = c("MML iw ", "VAE 10iw gs", "VAE 5iw gs", "VAE 1iw gs"))
  ) %>%
  ggplot(aes(y=lca_acc, x=method, col=method)) +
  facet_grid(nclass~nitems, scales = 'free_y', ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() +
  ylim(.9,1)

# runtime plot LCA
results %>% 
  filter((decay == .9 &  lr==.01&lcamethod!='vq')| est == 'MML', model=='LCA') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw ', lcamethod),
         method = factor(method, levels = c("MML iw ", "VAE 10iw gs", "VAE 5iw gs", "VAE 1iw gs"))
  ) %>%
  ggplot(aes(y=runtime, x=method, col=method)) +
  facet_grid(nclass~nitems, scales = 'free_y', ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() 

########################## GDINA ###################################
# itempars plot GDINA
results %>% 
  filter((decay == .9  &  lr==.01)| est == 'MML', model=='GDINA') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw'),
         method = factor(method, levels = c("MML iw", "VAE 10iw", "VAE 5iw", "VAE 1iw"))
  ) %>%
  filter(mse_itempars<0.01) %>%
  ggplot(aes(y=mse_itempars, x=method, col=method)) +
  facet_grid(nclass~nitems, scales = 'free_y') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() +
  ylim(0,0.01)

# acc plot GDINA
results %>% 
  filter((decay == .9 &  lr==.01)| est == 'MML', model=='GDINA') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw'),
         method = factor(method, levels = c("MML iw", "VAE 10iw", "VAE 5iw", "VAE 1iw"))
  ) %>%
  filter(lca_acc>.97 | (nitems==20&nclass=10)) %>%
  ggplot(aes(y=lca_acc, x=method, col=method)) +
  facet_grid(nclass~nitems, scales = 'free_y', ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() 

# runtime plot GDINA
results %>% 
  filter((decay == .9 &  lr==.01)| est == 'MML', model=='GDINA') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw'),
         method = factor(method, levels = c("MML iw", "VAE 10iw", "VAE 5iw", "VAE 1iw"))
  ) %>%
  ggplot(aes(y=runtime, x=method, col=method)) +
  facet_grid(nclass~nitems, scales = 'free_y', ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() 

results %>% 
  filter((decay == .9 &  lr==.01), n_iw_samples==| est == 'MML', model=='GDINA') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw'),
         method = factor(method, levels = c("MML iw", "VAE 10iw", "VAE 5iw", "VAE 1iw"))
  ) %>%
  ggplot(aes(x=runtime, col=n_iw_samples)) +
  facet_grid(nclass~nitems, scales = 'free_y', ) +
  geom_histogram()

results %>% 
  filter((decay == .9 &  lr==.01& n_iw_samples==1)| est == 'MML', model=='GDINA') %>%
  group_by(est) %>%
  summarise(mean(runtime))
  


######################### MIXIRT ############################
results %>% 
  filter((decay == .9&  lr==.01)|est=='MML', model=='MIXIRT') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw'),
         method = factor(method, levels = c("MML iw", "VAE 10iw", "VAE 5iw", "VAE 1iw"))
  ) %>%
  ggplot(aes(y=mse_intercepts, x=method, col=method)) +
  facet_grid(mirt_dim~., scales = 'free_y', ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() +
  ylim(0,0.05)


results %>% 
  filter((decay == .9 &  lr==.01)|est=='MML', model=='MIXIRT') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw'),
         method = factor(method, levels = c("MML iw", "VAE 10iw", "VAE 5iw", "VAE 1iw")),
         runtime = ifelse(est=='MML', runtime*3600, runtime) # convert hours to seconds for mplus
  ) %>%
  ggplot(aes(y=lca_acc, x=method, col=method)) +
  facet_grid(mirt_dim~., scales = 'free_y') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() 

results %>% 
  filter((decay == .9 &  lr==.01)|est=='MML', model=='MIXIRT') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw'),
         method = factor(method, levels = c("MML iw", "VAE 10iw", "VAE 5iw", "VAE 1iw")),
         runtime = ifelse(est=='MML', runtime*3600, runtime) # convert hours to seconds for mplus
  ) %>%
  ggplot(aes(y=runtime, x=method, col=method)) +
  facet_grid(mirt_dim~., scales = 'free_y') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() 


results %>% 
  filter((decay == .9 &  lr==.01)|est=='MML', model=='MIXIRT') %>%
  mutate(method=paste0(est,' ', n_iw_samples,'iw'),
         method = factor(method, levels = c("MML iw", "VAE 10iw", "VAE 5iw", "VAE 1iw")),
         runtime = ifelse(est=='MML', runtime*3600, runtime) # convert hours to seconds for mplus
  ) %>%
  ggplot(aes(y=mse_theta, x=method, col=method)) +
  facet_grid(mirt_dim~., scales = 'free_y') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text=element_text(size=20)) +
  geom_boxplot() +
  geom_hline(yintercept = 0.3363, linetype = "dashed", color = "black")



#############################################################
############# temp sim ######################################

results %>% 
  filter((decay == .9&  lr==.01, n_iw_samples==1)|est=='MML', model=='MIXIRT')


