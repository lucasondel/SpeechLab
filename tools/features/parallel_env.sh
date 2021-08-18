# How to call the parallel environment. In `run.sh`, the features
# extraction command will be called as:
#   $parallel_cmd <feaextract_cmd> [option] <arg1> <arg2>

parallel_cmd="qsub -sync y -N fea-extract -V -S /bin/bash -cwd -b y -j y -o $logdir -t 1:$njobs"
