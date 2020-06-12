
# check current task, the script would run the next round experiment when current pid is finished
pid=$1
script=train.sh
config=$2

if [ "$pid" != "" ]
then
  while true
  do
    nvidia-smi | grep $pid
    if [ $? -eq 0 ]; then sleep 1m; continue; else break; fi
  done
  #sleep 10
  echo "starting script"
  bash $script $config
fi

