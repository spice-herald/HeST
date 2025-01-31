line=$(sed -n "1p" ../parameters/height_sweep_0_5.txt)
INPUT_FILE=($line)
echo "${INPUT_FILE[0]}"