path = '~/Documents/GitHub/Discrete_VAEs/results/metrics/'
files = list.files(path)
model = amortized = nclass = replication = nitems = nrep = n_iw_samples =
  lr = decay = mintemp=lcamethod=lc_acc=mse_itempars=mse_a=mse_b=runtime= mirt_dim=c()
for (file in files ){
  split = strsplit(file, '_')[[1]]
  
  if (split[1] == 'mmlmetrics'){
    model = c(model, split[2])
    nclass = c(nclass, split[3])
    replication = c(replication, split[4])
    nitems = c(nitems, split[5])
    nrep = c(nrep, substr(split[6], 1, nchar(split[6])-4))
    n_iw_samples = c(n_iw_samples, '')
    lr = c(lr, '')
    decay = c(decay, '')
    mintemp = c(mintemp, '')
    lcamethod = c(lcamethod, '')
    mirt_dim = c(mirt_dim, '')
    amortized = c(amortized, 'False')
  } else if(split[1] == 'metrics'){
    replication = c(replication, split[2])
    model = c(model, split[3])
    n_iw_samples = c(n_iw_samples, split[4])
    lr = c(lr, split[5])
    decay = c(decay, split[6])
    mintemp = c(mintemp, split[7])
    nitems = c(nitems, split[8])
    amortized = c(amortized, split[9])
    lcamethod=c(lcamethod, substr(split[11], 1, nchar(split[11])-4))
    nrep = c(nrep, 1)
    if (split[3]=='MIXIRT'){
      mirt_dim=c(mirt_dim, substr(split[10], 1, nchar(split[10])-4))
      nclass = c(nclass, 2)
    } else if(split[3]=='LCA'){
      mirt_dim = c(mirt_dim, '')
      nclass = c(nclass, split[10])
    } else if(split[3]=='GDINA'){
      mirt_dim = c(mirt_dim, '')
      nclass=c(nclass, substr(split[10], 1, nchar(split[10])-4))
    }
    
  }else if(split[1]=='mplus'){
    replication = c(replication, substr(split[4], 1, nchar(split[4])-4))
    model = c(model, 'MIXIRT')
    n_iw_samples = c(n_iw_samples, '')
    lr = c(lr, '')
    decay = c(decay, '')
    mintemp = c(mintemp, '')
    amortized = c(amortized, '')
    mirt_dim = c(mirt_dim, 3)
    nitems = c(nitems, 28)
    nclass = c(nclass, 2)
    lcamethod=c(lcamethod, 'gs')
    nrep = c(nrep, split[3])
  }
  
  metrics = read.table(paste0(path, file))
  
  lc_acc = c(lc_acc, metrics$V1[1])
  mse_itempars = c(mse_itempars, as.numeric(metrics$V1[2]))
  if (split[3] == 'MIXIRT' | split[1]=='mplus'){
    mse_b = c(mse_b, metrics$V1[9])
    mse_a = c(mse_a, metrics$V1[10])
  } else{
    mse_b = c(mse_b, NA)
    mse_a = c(mse_a, NA)
  }
  runtime = c(runtime, metrics$V1[8])
}

results = data.frame(replication, model, n_iw_samples, lr, decay, 
                     mintemp, nitems, amortized, nclass, lcamethod, lc_acc, mse_itempars,
                     mse_a, mse_b,runtime, nrep, mirt_dim)


write.csv(results, '~/Documents/discrete_results.csv')

x = results  

unique(x[x$model=='GDINA',]$n_iw_samples)



options(device = "RStudioGD")
library(dplyr)
library(ggplot2)
p1 = results %>%
  mutate(method=paste(model, n_iw_samples, nrep)) %>%
  ggplot(aes(y=mse_itempars, x=method, col=method)) +
  geom_histogram()

ggsave(p1, 'test.png')

library(tidyverse)
results %>%
  filter(model == 'LCA', mse_itempars < 0.01) %>%
  mutate(method = paste(model, n_iw_samples, nrep, amortized)) %>%
  distinct(method)

results %>%
  filter(model == 'LCA') %>%
  mutate(method = paste(model, n_iw_samples, nrep, amortized),
         method = factor(method, levels = c("LCA 1 1 True", "LCA 5 1 True", "LCA 10 1 True",  'LCA 1 1 False', 'LCA  1 False', 'LCA  5 False')),
         method = factor(method, labels = c("GS-VAE 1iw", "GS-VAE 5iw", "GS-VAE 10iw",'standard VI', 'MML 1 Start', 'MML 5 Starts')),
         mse_itempars=as.numeric(mse_itempars)) %>%
  ggplot(aes(y = mse_itempars, x = method, group = method)) +  # Use method for x-axis
  geom_boxplot() +
  facet_grid(nclass ~ nitems, scales = 'free_y', labeller = labeller(nclass = function(x) paste(x, 'Classes'), nitems = function(x) paste(x, 'Items'))) +
  labs(x = "", y = "MSE") +  # Add axis labels
  theme_minimal() +  # Use a minimal theme to remove the gray background
  theme(
    axis.title = element_text(size = 25),  # Increase axis title text size
    axis.text.x = element_text(size = 20, angle = 45, hjust = 1),  # Rotate x-axis text for better readability
    axis.text.y = element_text(size = 20),  # Rotate x-axis text for better readability
    axis.text = element_text(size = 20),   # Increase axis text size
    strip.text = element_text(size = 20)   # Increase facet label text size
  )

results %>%
  distinct(n_iw_samples)

results %>%
  filter(model == 'LCA') %>%
  mutate(method = paste(model, n_iw_samples, nrep, amortized),
         method = factor(method, levels = c("LCA 1 1 True", "LCA 5 1 True", "LCA 10 1 True", 'LCA  1 False', 'LCA  5 False', 'LCA 1 1 False')),
         method = factor(method, labels = c("CLVM 1iw", "CLVM 5iw", "CLVM 10iw", 'MML 1 Start', 'MML 5 Starts', 'VI'))) %>%
  ggplot(aes(y = mse_itempars, x = method, group = method)) +  # Use method for x-axis
  geom_boxplot() +
  facet_grid(nclass ~ nitems, scales = 'free_y', labeller = labeller(nclass = function(x) paste(x, 'Attributes'), nitems = function(x) paste(x, 'Items'))) +
  labs(x = "", y = "MSE") +  # Add axis labels
  theme_minimal() +  # Use a minimal theme to remove the gray background
  theme(
    axis.title = element_text(size = 25),  # Increase axis title text size
    axis.text.x = element_text(size = 20, angle = 45, hjust = 1),  # Rotate x-axis text for better readability
    axis.text.y = element_text(size = 20),  # Rotate x-axis text for better readability
    axis.text = element_text(size = 20),   # Increase axis text size
    strip.text = element_text(size = 20)   # Increase facet label text size
  )



results %>%
  filter(model == 'GDINA') %>%
  mutate(method = paste(model, n_iw_samples, nrep, amortized),
         method = factor(method, levels = c("GDINA 1 1 True", "GDINA 5 1 True", "GDINA 10 1 True",  'GDINA 1 1 False', 'GDINA  1 False', 'GDINA  5 False')),
         method = factor(method, labels = c("GS-VAE 1iw", "GS-VAE 5iw", "GS-VAE 10iw",'standard VI', 'MML 1 Start', 'MML 5 Starts')),
         mse_itempars=as.numeric(mse_itempars)) %>%
  ggplot(aes(y = mse_itempars, x = method, group = method)) +  # Use method for x-axis
  geom_boxplot() +
  facet_grid(nclass ~ nitems, scales = 'free_y', labeller = labeller(nclass = function(x) paste(x, 'Classes'), nitems = function(x) paste(x, 'Items'))) +
  labs(x = "", y = "MSE") +  # Add axis labels
  theme_minimal() +  # Use a minimal theme to remove the gray background
  theme(
    axis.title = element_text(size = 25),  # Increase axis title text size
    axis.text.x = element_text(size = 20, angle = 45, hjust = 1),  # Rotate x-axis text for better readability
    axis.text.y = element_text(size = 20),  # Rotate x-axis text for better readability
    axis.text = element_text(size = 20),   # Increase axis text size
    strip.text = element_text(size = 20)   # Increase facet label text size
  )

results %>%
  filter(model == 'GDINA') %>%
  mutate(method = paste(model, n_iw_samples, nrep, amortized),
         method = factor(method, levels = c("GDINA 1 1 True", "GDINA 5 1 True", "GDINA 10 1 True", 'GDINA  1 False', 'GDINA  5 False', 'GDINA 1 1 False')),
         method = factor(method, labels = c("CLVM 1iw", "CLVM 5iw", "CLVM 10iw", 'MML 1 Start', 'MML 5 Starts', 'VI'))) %>%
  ggplot(aes(y = mse_itempars, x = method, group = method)) +  # Use method for x-axis
  geom_boxplot() +
  facet_grid(nclass ~ nitems, scales = 'free_y', labeller = labeller(nclass = function(x) paste(x, 'Attributes'), nitems = function(x) paste(x, 'Items'))) +
  labs(x = "", y = "MSE") +  # Add axis labels
  theme_minimal() +  # Use a minimal theme to remove the gray background
  theme(
    axis.title = element_text(size = 25),  # Increase axis title text size
    axis.text.x = element_text(size = 20, angle = 45, hjust = 1),  # Rotate x-axis text for better readability
    axis.text.y = element_text(size = 20),  # Rotate x-axis text for better readability
    axis.text = element_text(size = 20),   # Increase axis text size
    strip.text = element_text(size = 20)   # Increase facet label text size
  )

results %>%
  filter(model == 'MIXIRT') %>%
  mutate(method = paste(model, n_iw_samples, nrep, amortized)) %>%
  distinct(method)


results %>%
  filter(model == 'MIXIRT', mse_b<.05) %>%
  mutate(method = paste(model, n_iw_samples, nrep, amortized),
         method = factor(method, levels = c("MIXIRT 1 1 True", "MIXIRT 5 1 True", "MIXIRT 10 1 True", 'MIXIRT 1 1 False', 'MIXIRT 5 1 False', 'MIXIRT 10 1 False', 'MIXIRT  1 ', 'MIXIRT  5 ')),
         method = factor(method, labels = c("CLVM 1iw", "CLVM 5iw", "CLVM 10iw", "VI 1iw", "VI 5iw", "VI 10iw", "MML 1 Start", "MML 5 Starts"))) %>%
  ggplot(aes(y = mse_b, x = method, group = method)) +  # Use method for x-axis
  geom_boxplot() +
  facet_grid(nclass ~ nitems, scales = 'free_y', labeller = labeller(nclass = function(x) paste(x, 'Attributes'), nitems = function(x) paste(x, 'Items'))) +
  labs(x = "", y = "MSE") +  # Add axis labels
  theme_minimal() +  # Use a minimal theme to remove the gray background
  theme(
    axis.title = element_text(size = 25),  # Increase axis title text size
    axis.text.x = element_text(size = 20, angle = 45, hjust = 1),  # Rotate x-axis text for better readability
    axis.text.y = element_text(size = 20),  # Rotate x-axis text for better readability
    axis.text = element_text(size = 20),   # Increase axis text size
    strip.text = element_text(size = 20)   # Increase facet label text size
  )


results %>%
  filter(model == 'MIXIRT', mse_b<.05) %>%
  mutate(method = paste(model, n_iw_samples, nrep, amortized),
         method = factor(method, levels = c("MIXIRT 1 1 True", "MIXIRT 5 1 True", "MIXIRT 10 1 True", 'MIXIRT 1 1 False', 'MIXIRT 5 1 False', 'MIXIRT 10 1 False', 'MIXIRT  1 ', 'MIXIRT  5 ')),
         method = factor(method, labels = c("CLVM 1iw", "CLVM 5iw", "CLVM 10iw", "VI 1iw", "VI 5iw", "VI 10iw", "MML 1 Start", "MML 5 Starts"))) %>%
  ggplot(aes(y = runtime, x = method, group = method)) +  # Use method for x-axis
  geom_boxplot() +
  facet_grid(nclass ~ nitems, scales = 'free_y', labeller = labeller(nclass = function(x) paste(x, 'Attributes'), nitems = function(x) paste(x, 'Items'))) +
  labs(x = "", y = "MSE") +  # Add axis labels
  theme_minimal() +  # Use a minimal theme to remove the gray background
  theme(
    axis.title = element_text(size = 25),  # Increase axis title text size
    axis.text.x = element_text(size = 20, angle = 45, hjust = 1),  # Rotate x-axis text for better readability
    axis.text.y = element_text(size = 20),  # Rotate x-axis text for better readability
    axis.text = element_text(size = 20),   # Increase axis text size
    strip.text = element_text(size = 20)   # Increase facet label text size
  )



results %>%
  filter(model == 'MIXIRT', mse_b<.05) %>%
  mutate(method = paste(model, n_iw_samples, nrep, amortized),
         method = factor(method, levels = c("MIXIRT 1 1 True", "MIXIRT 5 1 True", "MIXIRT 10 1 True", 'MIXIRT 1 1 False', 'MIXIRT 5 1 False', 'MIXIRT 10 1 False', 'MIXIRT  1 ', 'MIXIRT  5 ')),
         method = factor(method, labels = c("CLVM 1iw", "CLVM 5iw", "CLVM 10iw", "VI 1iw", "VI 5iw", "VI 10iw", "MML 1 Start", "MML 5 Starts"))) %>%
  ggplot(aes(y = mse_itempars, x = method, group = method)) +  # Use method for x-axis
  geom_boxplot() +
  facet_grid(nclass ~ nitems, scales = 'free_y', labeller = labeller(nclass = function(x) paste(x, 'Classes'), nitems = function(x) paste(x, 'Items'))) +
  labs(x = "", y = "MSE") +  # Add axis labels
  theme_minimal() +  # Use a minimal theme to remove the gray background
  theme(
    axis.title = element_text(size = 25),  # Increase axis title text size
    axis.text.x = element_text(size = 20, angle = 45, hjust = 1),  # Rotate x-axis text for better readability
    axis.text.y = element_text(size = 20),  # Rotate x-axis text for better readability
    axis.text = element_text(size = 20),   # Increase axis text size
    strip.text = element_text(size = 20)   # Increase facet label text size
  )

results %>%
  filter(model == 'MIXIRT') %>%
  mutate(method = paste(model, n_iw_samples, nrep, amortized),
         method = factor(method, levels = c("MIXIRT 1 1 True", "MIXIRT 5 1 True", "MIXIRT 10 1 True", 'MIXIRT 1 1 False', 'MIXIRT 5 1 False', 'MIXIRT 10 1 False', 'MIXIRT  1 ', 'MIXIRT  5 ')),
         method = factor(method, labels = c("CLVM 1iw", "CLVM 5iw", "CLVM 10iw", "VI 1iw", "VI 5iw", "VI 10iw", "MML 1 Start", "MML 5 Starts")),
         runtime=as.numeric(runtime )) %>%
  ggplot(aes(y = runtime, x = method, group = method)) +  # Use method for x-axis
  geom_boxplot() +
  facet_grid(nclass ~ nitems, scales = 'free_y', labeller = labeller(nclass = function(x) paste(x, 'Classes'), nitems = function(x) paste(x, 'Items'))) +
  labs(x = "", y = "MSE") +  # Add axis labels
  theme_minimal() +  # Use a minimal theme to remove the gray background
  theme(
    axis.title = element_text(size = 25),  # Increase axis title text size
    axis.text.x = element_text(size = 20, angle = 45, hjust = 1),  # Rotate x-axis text for better readability
    axis.text.y = element_text(size = 20),  # Rotate x-axis text for better readability
    axis.text = element_text(size = 20),   # Increase axis text size
    strip.text = element_text(size = 20)   # Increase facet label text size
  )


results %>%
  filter(model=='MIXIRT', lr!='') %>%
  group_by(n_iw_samples, amortized) %>%
  summarize(mean(as.numeric(runtime)))
