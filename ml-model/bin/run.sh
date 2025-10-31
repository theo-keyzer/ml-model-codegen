run="go run ../src/main.go ../src/gen.go ../src/run.go ../src/structs.go  ../src/collect.go ../src/hcon4.go"

#$run  ../actors/moe2.act  ../nets/kernel2.net,../nets/moe.net
#if [ $? != 0 ]; then echo moe2.act  moe.net has errors; fi

$run  ../actors/net3.act  ../nets/res50_2.net 
if [ $? != 0 ]; then echo net3.act res50_2.net has errors; fi


