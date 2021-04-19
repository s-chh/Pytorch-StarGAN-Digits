# Pytorch-StarGAN-Digits
Unofficial Pytorch implementation of StarGAN for generating Digit-5 datasets (MNIST, SVHN, SynDigits, MNIST-M, and USPS).

$ ./tree-md .
# Project tree

~> tree -d /proc/self/
/proc/self/
|-- attr
|-- cwd -> /proc
|-- fd
|   `-- 3 -> /proc/15589/fd
|-- fdinfo
|-- net
|   |-- dev_snmp6
|   |-- netfilter
|   |-- rpc
|   |   |-- auth.rpcsec.context
|   |   |-- auth.rpcsec.init
|   |   |-- auth.unix.gid
|   |   |-- auth.unix.ip
|   |   |-- nfs4.idtoname
|   |   |-- nfs4.nametoid
|   |   |-- nfsd.export
|   |   `-- nfsd.fh
|   `-- stat
|-- root -> /
`-- task
    `-- 15589
        |-- attr
        |-- cwd -> /proc
        |-- fd
        | `-- 3 -> /proc/15589/task/15589/fd
        |-- fdinfo
        `-- root -> /

27 directories

## Generated Samples
Input | SynDigits | MNIST | MNIST-M | SVHN | USPS 
--- | --- | --- | --- | --- | ---
![Input](/Results/Input.png) | ![SynDigits](/Results/SynDigits.png) | ![MNIST](/Results/MNIST.png) | ![MNIST-M](/Results/MNISTM.png) | ![SVHN](/Results/SVHN.png) | ![USPS](/Results/USPS.png) 

<img src="/Results/Digits.png" width="500"></img>
