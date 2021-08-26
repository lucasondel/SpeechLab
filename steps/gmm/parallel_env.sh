parallel_cmd_posterior="qsub -sync y -N gmm-posterior -V -S /bin/bash -cwd -b y -j y -t 1:$njobs"
parallel_cmd_predict="qsub -sync y -N gmm-predict -V -S /bin/bash -cwd -b y -j y -t 1:$njobs"
