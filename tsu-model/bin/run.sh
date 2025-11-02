run="go run ../src/main.go ../src/gen.go ../src/run.go ../src/structs.go  ../src/collect.go ../src/hcon4.go"

$run  ../actors/jax_ising_pgm.act  ../nets/ising_pgm.net
if [ $? != 0 ]; then echo  jax_ising_pgm.act  ising_pgm.net has errors; fi


