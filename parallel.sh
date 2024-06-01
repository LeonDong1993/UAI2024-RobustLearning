set -e

if [ "$#" -lt 2 ]; then
    echo "Error: Insufficient number of inputs, give me codename, src notebook"
    exit 1
fi

snapshot=$2
mkdir ./results/$1/
cp $snapshot ./results/$1/

for name in mnist airquality parkinson energy sdd hepmass onlinenews superconduct miniboone
do   
    bash run.sh $1 $name $2 &>/dev/null &
    sleep 3
done

wait

