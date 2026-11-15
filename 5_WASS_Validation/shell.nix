# Development shell for WASS on NixOS.
#
#   cd 5_WASS_Validation
#   ./build_wass.sh     # once (and after updating the wass submodule): builds the C++ pipeline into wass_bin/
#   nix-shell           # then: wass_prepare / wass_match / wass_autocalibrate / wass_stereo,
#                       #       wasscli, wassgridsurface and the viewers' Python deps are all available
#
# Build dependencies come from nixpkgs. boost187 rather than `boost`: Boost 1.89 removed the boost_system
# library that WASS's CMake asks for. The WASS Python tools are built from the submodule's own source.
{ pkgs ? import <nixpkgs> { } }:

let
  py = pkgs.python3.pkgs;

  wassPythonPackage = pname: src: deps: py.buildPythonPackage {
    inherit pname src;
    version = "submodule";
    pyproject = true;
    build-system = [ py.setuptools py.setuptools-scm ];
    dependencies = deps;
    pythonRemoveDeps = [ "opencv-python" ]; # provided by nixpkgs' opencv4 (same cv2 module, different dist name)
    doCheck = false;
  };

  wasscli = wassPythonPackage "wasscli" ./wass/cli
    (with py; [ tqdm inquirerpy colorama numpy ]);

  wassgridsurface = wassPythonPackage "wassgridsurface" ./wass/gridding
    (with py; [ torch matplotlib colorama scipy numpy tqdm opencv4 netcdf4 h5py ]);

  python = pkgs.python3.withPackages (ps: [
    wasscli
    wassgridsurface
    ps.numpy
    ps.netcdf4
    ps.opencv4
    ps.pygame-ce # view_stereo.py / view_nc.py live window
    # WASSfast (CNN mode), run from the wassfast/ submodule source (see `wassfast` below).
    # Its pyproject.toml only packages the top-level module (no cnn/, cvproc/ or model weights), so it isn't pip-built.
    ps.tensorflow-bin # prebuilt TF 2.21 wheel; wassfast pins 2.20
    ps.keras # tensorflow.keras is Keras 3; tensorflow-bin doesn't pull it in
    ps.scipy
    ps.matplotlib
    ps.tqdm
    ps.mako
    ps.h5py
    ps.python-dateutil
    ps.protobuf
    ps.psutil # run_wassfast.py: CPU time / peak memory
  ]);

  # `wassfast` command: the submodule's own code, so its relative model-weight paths resolve.
  wassfast = pkgs.writeShellScriptBin "wassfast" ''
    export PYTHONPATH="${toString ./wassfast}''${PYTHONPATH:+:$PYTHONPATH}"
    exec ${python}/bin/python3 -m wassfast "$@"
  '';
in
pkgs.mkShell {
  packages = [ python wassfast ] ++ (with pkgs; [ cmake gnumake gcc pkg-config opencv4 boost187 lapack blas ffmpeg ]);

  shellHook = ''
    export WASS_VALIDATION_DIR="${toString ./.}"
    export PATH="$WASS_VALIDATION_DIR/wass_bin/bin:$PATH"
    export TF_CPP_MIN_LOG_LEVEL=1 # hide TensorFlow's INFO spam
    if [ ! -x "$WASS_VALIDATION_DIR/wass_bin/bin/wass_stereo" ]; then
      echo "WASS executables not built yet: run ./build_wass.sh (outside this shell)."
    fi
  '';
}
