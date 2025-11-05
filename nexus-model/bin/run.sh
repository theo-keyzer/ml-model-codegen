run="go run ../src/main.go ../src/gen.go ../src/run.go ../src/structs.go  ../src/collect.go ../src/hcon4.go"

$run  ../actors/cnn.act  ../nets/cnn.net   >cnn.out
if [ $? != 0 ]; then echo  cnn.act  cnn.net has errors; fi

$run  ../actors/snn.act  ../nets/snn.net >snn.out
if [ $? != 0 ]; then echo snn.act  snn.net  has errors; fi

