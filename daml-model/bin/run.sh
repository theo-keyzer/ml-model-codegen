run="go run ../src/main.go ../src/gen.go ../src/run.go ../src/structs.go ../src/collect.go ../src/hcon4.go"

$run  ../actors/nas-darts.act  ../nets/nas-dart.daml >nas-dart.out
if [ $? != 0 ]; then echo nas-darts.act  nas-dart.daml has errors; fi


