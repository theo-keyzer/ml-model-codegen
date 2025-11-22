run="go run ../src/main.go ../src/gen.go ../src/run.go ../src/structs.go ../src/collect.go ../src/rio.go"

$run  ../actors/gen-json.act  ../inputs/pbit-maxcut.rio >pbit-maxcut.json
if [ $? != 0 ]; then echo gen-json.act pbit-maxcut.rio has errors; fi

$run  ../actors/pbit-maxcut.act ../inputs/pbit-maxcut.rio >pbit-maxcut.py
if [ $? != 0 ]; then echo pbit-maxcut.act pbit-maxcut.rio has errors; fi

