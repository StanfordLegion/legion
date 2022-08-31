# Jupyter Notebook

This directory contains the Jupyter notebook support for
the Python binding of Legion as well as the derived
libraries including Legate, cuNumeric and Pygion.
It provides the capability of running these libraries using
the jupyter notebook from any browers. 

## Quick Start
### Pre-requisite
* Python >= 3.6
* [Python binding of Legion](https://github.com/StanfordLegion/legion/tree/stable/bindings/python) 
  or [Legate](https://github.com/nv-legate/legate.core) needs to be installed.
* Install Jupyter notebook

        pip install notebook

### Install the Legion IPython kernel
```
python ./install.py --(configurations)
```
Please refer to the [IPython Kernel Configurations](#kernel-configurations) section for the configuration details.

If the installation is successed, the following log will be printed to the terminal.
The `legion_kernel_nocr` is the IPython kernel name, and the `Legate_SM_GPU` is the display name
of the kernel, which can be modified by the configuration json file. 
`Legate` is the name entry in the json file, `SM` means the IPython kernel
is only for shared memory machine, and `GPU` means GPU execution is enabled. 
```
IPython kernel: legion_kernel_nocr(Legate_SM_GPU) has been installed
```
The installed IPython kernel can be also seen by using the following command:
```
jupyter kernelspec list
```

### Create a turnel (Optional)
If you want to run the jupyter notebook server on a remote compute node instead of localhost, 
you can create a turnel from localhost to the compute node.
```
ssh -4 -t -L 8888:localhost:8002 username@login-node-hostname ssh -t -L 8002:localhost:8888 computing_node
```

### Start the Jupyter Notebook server
Launch jupyter notebook server on the compute node or localhost if the turnel is not created
```
jupyter notebook --port=8888 --no-browser
```

### Use the Jupyter Notebook in the browser
* Open the browser, type the addredd http://localhost:8888/?token=xxx, the token will be
displayed in the terminal once the server is started. 
* Once the webpage is loaded, click "New" on the right top corner, and click the kernel 
just installed. It is shown as the display name of the kernel, e.g. `Legate_SM_GPU`.

### Uninstall the IPython kernel
```
jupyter kernelspec uninstall legion_kernel_nocr
```
If the IPython kernel is re-installed, the old one will be automatically uninstalled by the install.py

## IPython Kernel Configurations
The IPython kernel can be configured by either passing arguments to `install.py` or using a json file.
The accepted arguments can be listed with 
```
python ./install_jupyter.py --help
```

It is always preferred to use a json file. 
The `legate.json` and `legion_python.json` are two templates respect to the legate wrapper and
legion_python. Most entries are using the following format:
```
"cpus": {
    "cmd": "--cpus",
    "value": 1
}
```
* `cpus` is the name of the field. 

* `cmd` is used to tell how to pass the value to the field.
For example, the legate wrapper uses `--cpus` to set the number of cpus, so the `cmd` in `legate.json`
is `--cpus`, while legion takes `-ll:cpu`, so the cmd in `legion_python.json` is `-ll:cpu`.

* `value` is the value of the field. It can be set to `null`. In this case, the value is read
from the command line arguments. 

Other configuration options can be added by either appending them to the command line arguments or
using the `other_options` field of the json file. 

## Magic Command
We provide a Jupyter magic command to display the IPython kernel configuration.
```
%load_ext legate.info
%legate_info
Number of CPUs to use per rank: 4
Number of GPUs to use per rank: 1
Number of OpenMP groups to use per rank: 0
Number of threads per OpenMP group: 4
Number of Utility processors per rank: 2
Amount of DRAM memory per rank (in MBs): 4000
Amount of DRAM memory per NUMA domain per rank (in MBs): 0
Amount of framebuffer memory per GPU (in MBs): 4000
Amount of zero-copy memory per rank (in MBs): 32
Amount of registered CPU-side pinned memory per rank (in MBs): 0
Number of nodes to use: 1
```  
