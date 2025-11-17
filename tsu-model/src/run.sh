run="/home/theo/Downloads/go1.24.5.linux-amd64/go/bin/go run main.go gen.go run.go structs.go  collect.go hcon4.go"

$run  jax_ising_pgm.act  ising_pgm.net >x1
if [ $? != 0 ]; then echo  jax_ising_pgm.act  ising_pgm.net has errors; fi

$run  max-cut.act  max-cut.net >max-cut.out
if [ $? != 0 ]; then echo  max-cut.act  max-cut.net has errors; fi

$run  max-cut-auto.act  max-cut.net,max-cut-auto.net >max-cut-auto.out
if [ $? != 0 ]; then echo  max-cut-auto.act  max-cut.net,max-cut-auto.net has errors; fi

#$run  p-bit.act  p-bit.net 
#if [ $? != 0 ]; then echo p-bit.act  p-bit.net has errors; fi

#$run  xx3.act  pbit-maxcut.net 
#if [ $? != 0 ]; then echo pbit-maxcut.act  pbit-maxcut.net has errors; fi

