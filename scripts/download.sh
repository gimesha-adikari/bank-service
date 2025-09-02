# put everything straight into Conda’s cache
PKGS="$HOME/miniconda3/pkgs"
CF="https://conda.anaconda.org/conda-forge"
mkdir -p "$PKGS" && cd "$PKGS"

# (you already have most of these; re-running is harmless)
wget -c "$CF/linux-64/onnxruntime-1.22.0-py311h4107d7c_200_cuda.conda"
wget -c "$CF/noarch/cuda-cudart_linux-64-12.6.77-h3f2d84a_0.conda"
wget -c "$CF/linux-64/cuda-cudart-12.6.77-h5888daf_0.conda"
wget -c "$CF/linux-64/cuda-nvrtc-12.6.85-hbd13f7d_0.conda"         # ← add this
wget -c "$CF/noarch/cuda-version-12.6-h7480c83_3.conda"            # ← add this (meta pin)

wget -c "$CF/linux-64/libcudnn-9.12.0.46-hf7e9902_0.conda"
wget -c "$CF/linux-64/libcudnn-dev-9.12.0.46-h58dd1b1_0.conda"
wget -c "$CF/linux-64/libcublas-12.6.4.1-h5888daf_1.conda"
wget -c "$CF/linux-64/libcufft-11.3.0.4-hbd13f7d_0.conda"
wget -c "$CF/linux-64/libcurand-10.3.7.77-hbd13f7d_0.conda"
wget -c "$CF/linux-64/libcusparse-12.5.4.2-hbd13f7d_0.conda"
wget -c "$CF/linux-64/libnvjitlink-12.9.86-h5888daf_1.conda"
