run="go run ../src/main.go ../src/gen.go ../src/run.go ../src/structs.go  ../src/collect.go ../src/hcon4.go"

$run  ../actors/jax_ising_pgm.act  ../nets/ising_pgm.net >ising_pgm.out
if [ $? != 0 ]; then echo  jax_ising_pgm.act  ising_pgm.net has errors; fi

$run  ../actors/max-cut.act  ../nets/max-cut.net >max-cut.out
if [ $? != 0 ]; then echo  max-cut.act  max-cut.net has errors; fi

$run  ../actors/max-cut-auto.act  ../nets/max-cut.net,../nets/max-cut-auto.net >max-cut-auto.out
if [ $? != 0 ]; then echo  max-cut-auto.act  max-cut.net,max-cut-auto.net has errors; fi

