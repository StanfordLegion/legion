This is an example of using Xilinx FPGAs as accelators in Realm.
===============	
## Prerequisite:
Xilinx XRT 2021.1 and Xilinx Vitis 2021.1.

This example is tested on an F1 node on AWS. Please refer to `https://github.com/aws/aws-fpga` for more information about using FPGA on AWS.
 
## Steps:
1. Create an AWS instance with FPGA Developer AMI (1.11.0 tested) in AWS Marketplace

2. Get AWS FGPA Development Kit: 
```
    git clone https://github.com/aws/aws-fpga.git $AWS_FPGA_REPO_DIR
```

3. Set up environment:
```
    source /home/centos/src/project_data/aws-fpga/vitis_setup.sh
    source /opt/Xilinx/Vivado/2021.1/settings64.sh

    export LG_RT_DIR=/home/centos/programs/legion_fpga/runtime
```

4. Build and run the example
```
    make build
    make run_fpga_vadd
```

5. Clean
```
    make clean
```
