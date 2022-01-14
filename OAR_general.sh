#!/bin/bash
#
#OAR -n mist2-tomo
#OAR -O logs/stdout.%jobid%
#OAR -E logs/stderr.%jobid%
#OAR -l nodes=1/core=8,walltime=12
#OAR -p not host like 'hib3-3301'

TOMOSCRIPT=$1
INIFILEPATH=$2
STUDYCASE=$3
SLICESIZE=$4

echo "Case is $STUDYCASE with $SLICESIZE where is $INIFILEPATH"

echo "Calling script $TOMOSCRIPT with python3"
if [ $SLICESIZE = 0 ]; then
	echo "Calculating projection 0"
	cmd0="python3 $TOMOSCRIPT $INIFILEPATH $STUDYCASE 0 0"
	eval $cmd0
elif [ $SLICESIZE -lt 0 ]; then
	echo "Calculating remaining projections"
	cmdEnd="python3 $TOMOSCRIPT $INIFILEPATH $STUDYCASE $5 $6"
	print $cmdEnd
	eval $cmdEnd
else
	echo "OAR_ARRAY_INDEX : $OAR_ARRAY_INDEX"
	cmdArray="python3 $TOMOSCRIPT $INIFILEPATH $STUDYCASE $(((OAR_ARRAY_INDEX-1)*SLICESIZE+1)) $((OAR_ARRAY_INDEX*SLICESIZE))"
	#echo $cmdArray
	eval $cmdArray
	echo ..done
fi

## ##OAR -t besteffort

