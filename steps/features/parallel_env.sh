rm -f $logdir/fea-extract*
parallel_cmd="qsub -sync y -N fea-extract -V -S /bin/bash -cwd -b y -j y -o $logdir -t 1:$njobs"
