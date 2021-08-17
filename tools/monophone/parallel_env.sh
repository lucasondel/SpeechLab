parallel_cmd_alis="qsub -sync y -N make-alis -V -S /bin/bash -cwd -b y -j y -o $logdir -t 1:$njobs"
