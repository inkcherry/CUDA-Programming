export PATH=$PATH:/usr/local/cuda-11.3/bin
alias cuc='nvcccompile(){ nvcc $1 -o cu.out; echo $1;};nvcccompile'
alias cur='./cu.out'
alias nvperf='nvprof ./cu.out'
# export LD_LIBYARY_PATH=
