library(MplusAutomation)
library(MASS)
library(abind)




theta_true = as.matrix(read.csv(paste0('~/Documents/GitHub/VAE_MIXIRT/true/pars/theta_', mirt_dim, '_', cov, '.csv')))
a0_true = read.csv(paste0('~/Documents/GitHub/VAE_MIXIRT/true/pars/slopes0_', mirt_dim, '_', cov, '.csv'))
a1_true = read.csv(paste0('~/Documents/GitHub/VAE_MIXIRT/true/pars/slopes1_', mirt_dim, '_', cov, '.csv'))
a_true = abind(a0_true, a1_true, along=3)
b_true = as.matrix(read.csv(paste0('~/Documents/GitHub/VAE_MIXIRT/true/pars/difficulty_', mirt_dim, '_', cov, '.csv')))
cl_true = read.csv(paste0('~/Documents/GitHub/VAE_MIXIRT/true/pars/class_', mirt_dim, '_', cov, '.csv'))
nitems = nrow(b_true)
for (iteration in 1:10){
  data = read.csv(paste0('~/Documents/GitHub/VAE_MIXIRT/true/data/data_', mirt_dim, '_', cov, '_', iteration, '.csv'))

  write.table(data,"mix_sim.dat",col=F,row=F)
  
  t1 = Sys.time()
  system(paste0("mplus mixIRT_", mirt_dim, "D.inp"))
  runtime = runtime = as.numeric(Sys.time()-t1,units="secs")
  
  res=readModels(paste0("mixIRT_", mirt_dim, "D.out"))
  pars = res$parameters$unstandardized
  
  
  nclass = 2
  b_est = matrix(nrow=nitems, ncol=nclass)
  for (c in 1:nclass){
    b_est[,c] = -pars$est[pars$paramHeader=='Thresholds' & pars$LatentClass==c]
  }
  a_est = array(0,dim=c(nitems,mirt_dim, nclass))
  for (item in 1:nitems){
    for (dim in 1:mirt_dim){
      for (cl in 1:nclass){
        estimate = pars$est[pars$paramHeader==paste0('F',dim, '.BY') &
                              pars$param == paste0('Y', item) &
                              pars$LatentClass ==cl]
        if (length(estimate) > 0){
          a_est[item, dim, cl] = estimate
        }
      }
      
    }
  }
  
  FS=as.matrix(read.table("FS.sav"))
  theta_est = FS[,(nitems+1):(nitems+mirt_dim), drop=FALSE]
  cl_est = FS[,ncol(FS)]-1
  
  # deal with label switching
  if (cor(cl_true, cl_est)<0){
    tmp = cl_est
    cl_est[tmp==0] = 1 
    cl_est[tmp==1] = 0
    
    b_est = b_est[,c(2,1)]
    a_est = a_est[,,c(2,1)]
  }
  # deal with inverted scales
  for (dim in 1:mirt_dim){
    if (cor(theta_true[,dim], theta_est[,dim]) < 0){
      theta_est[,dim] = -1 * theta_est[,dim]
      a_est[,dim,] = -1 * a_est[,dim,]
    }
  }
  
  acc = mean(cl_true==cl_est)
  msea = mean((a_est[a_true!=0]-a_true[a_true!=0])^2)
  mseb = mean((as.vector(b_est)-as.vector(b_true))^2)
  msetheta = mean((theta_est-theta_true)^2)
  
  fileConn<-file(paste0(c('~/Documents/GitHub/VAE_MIXIRT/results/metrics/mplus_', mirt_dim, '_', cov, '_', iteration, '.txt'), collapse=''))
  writeLines(c(as.character(msea), as.character(msetheta), as.character(mseb), as.character(acc), as.character(runtime)),fileConn)
  close(fileConn)
}

  }
}



write.table(data,"mix_sim.dat",col=F,row=F)

t1 = Sys.time()
system(paste0("mplus mixIRT_", mirt_dim, "D.inp"))
runtime = Sys.time() - t1

res=readModels(paste0("mixIRT_", mirt_dim, "D.out"))
pars = res$parameters$unstandardized


mirt_dim=10
nclass = 2
nitems = 110
b_est = matrix(nrow=nitems, ncol=nclass)
for (c in 1:nclass){
  b_est[,c] = -pars$est[pars$paramHeader=='Thresholds' & pars$LatentClass==c]
}
a_est = array(0,dim=c(nitems,mirt_dim, nclass))
for (item in 1:nitems){
  for (dim in 1:mirt_dim){
    for (cl in 1:nclass){
      estimate = pars$est[pars$paramHeader==paste0('F',dim, '.BY') &
                                    pars$param == paste0('Y', item) &
                                    pars$LatentClass ==cl]
      if (length(estimate) > 0){
        a_est[item, dim, cl] = estimate
      }
    }
    
  }
}
plot(a0_true[a0_true!=0], a_est[,,1][a0_true!=0])

plot(b_true[,1], b_est[,1])
abline(0,1)

eta_est=read.table("FS.sav")

pdf(file='~/Rplot.pdf')
par(mfrow=c(3,3))
for(f in 1:nf){
  plot(theta_true[,f],eta_est[,nit+f])
  abline(0,1,col="red")
  plot(a1_true[,f],a_est[,f])
  abline(0,1,col="red")
}
for(c in 1:nc)
  plot(b[,c],b1_est[,c])
  abline(0,1)
    
dev.off()


