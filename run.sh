set -e

if [ "$#" -lt 3 ]; then
    echo "Error: Insufficient number of inputs, give me codename and datasetname, src notebook."
    exit 1
fi


# create snapshot 
snapshot=$3
logfile="./results/$1/$2_exp.log"
name=$2

#for name in mnist airquality parkinson energy sdd hepmass onlinenews superconduct miniboone
#do   
outname="$1_$name"
papermill $snapshot $outname.ipynb -p data_name $name -p testing 0 -p log_file $logfile -p G_delta "0.5"
nbsave $outname.ipynb $outname
mv $outname.html ./results/$1/
rm $outname.ipynb
# done


