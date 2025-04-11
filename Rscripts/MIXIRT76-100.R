library(MplusAutomation)
library(MASS)
library(abind)

# read command line arguments
#args = commandArgs(trailingOnly = FALSE)
MIRT_DIM = 3
NITEMS = ifelse(MIRT_DIM==3, 28, 110)

setwd(getSrcDirectory(function(){})[1])
for (replication in 76:100){
  for (nrep in c(1,5)){
    print(paste(c('replication', replication)))
    data = np$load(path.expand(paste0(c('~/Downloads/saved_data_snellius/MIXIRT/data/', MIRT_DIM, '_', replication, '.npy'), collapse = ''))) 
    theta_true = np$load(path.expand(paste0(c('~/Downloads/saved_data_snellius/MIXIRT/theta/', MIRT_DIM,'.npy'), collapse = ''))) 
    itempars_true = np$load(path.expand(paste0(c('~/Downloads/saved_data_snellius/MIXIRT/itempars/', MIRT_DIM,'.npy'), collapse = ''))) 
    class_true = np$load(path.expand(paste0(c('~/Downloads/saved_data_snellius/MIXIRT/class/', MIRT_DIM,'.npy'), collapse = ''))) 
    cl_true = apply(class_true,1, which.max)-1
    
    write.table(data,"./Mplus/mix_sim.dat",col=F,row=F)
    
    t1 = Sys.time()
    system(paste0("mplus ./Mplus/mixIRT_", MIRT_DIM, "D_", nrep, "START.inp"))
    runtime = runtime = as.numeric(Sys.time()-t1,units="secs")
    
    res=readModels(paste0("mixIRT_", MIRT_DIM, "D.out"))
    pars = res$parameters$unstandardized
    
    
    nclass = 2
    b_est = matrix(nrow=NITEMS, ncol=nclass)
    for (c in 1:nclass){
      b_est[,c] = -pars$est[pars$paramHeader=='Thresholds' & pars$LatentClass==c]
    }
    a_est = array(0,dim=c(NITEMS,MIRT_DIM, nclass))
    for (item in 1:NITEMS){
      for (dim in 1:MIRT_DIM){
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
    
    FS=as.matrix(read.table("./Mplus/FS.sav"))
    theta_est = FS[,(NITEMS+1):(NITEMS+MIRT_DIM), drop=FALSE]
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
    for (dim in 1:MIRT_DIM){
      if (cor(theta_true[,dim], theta_est[,dim]) < 0){
        theta_est[,dim] = -1 * theta_est[,dim]
        a_est[,dim,] = -1 * a_est[,dim,]
      }
    }
    
    
    
    tmp = b_est
    dim(tmp) = c(28,1,2)
    itempars_est = abind(tmp, a_est, along=2)
    
    acc = mean(cl_true==cl_est)
    a_true = itempars_true[,2:4,]
    b_true = itempars_true[,1,]
    msea = mean((a_est[a_true!=0]-a_true[a_true!=0])^2)
    mseb = mean((as.vector(b_est)-as.vector(b_true))^2)
    biasa = mean(a_est[a_true!=0]-a_true[a_true!=0])
    biasb = mean(as.vector(b_est)-as.vector(b_true))
    mse_itempars = mean((itempars_true[itempars_true!=0]- itempars_est[itempars_true!=0])^2)
    msetheta = mean((theta_est-theta_true)^2)
    
    bias_itempars = mean(itempars_est[itempars_true!=0]- itempars_true[itempars_true!=0])
    bias_theta = mean(theta_est- theta_true)
    
    
    fileConn<-file(paste0(c('./results/metrics/mplus_', MIRT_DIM, '_', nrep, '_', replication,'.txt'), collapse=''))
    writeLines(c(as.character(acc), as.character(mse_itempars), as.character(msetheta), 
                 as.character(var(itempars_est)), as.character(var(as.vector(theta_est))), 
                                                               as.character(bias_itempars), as.character(bias_theta), as.character(runtime), 
                                                               as.character(mseb), as.character(msea), as.character(biasb), as.character(biasa)),fileConn)
               close(fileConn)
  }
  
}

