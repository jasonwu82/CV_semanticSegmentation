### Environment Installment

module load python

python -V
- If tensorflow env already exist in conda

  conda remove --name tensorflow --all

  conda create -n tensorflow python=3.5

  source activate tensorflow
- Version: Ubuntu/Linux 64-bit, CPU only, Python 3.5

  export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp35-cp35m-linux_x86_64.whl

  pip install --ignore-installed --upgrade $TF_BINARY_URL
