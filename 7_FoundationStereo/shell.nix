# Development shell for FoundationStereo on NixOS (WSL2).
#
#   cd 7_FoundationStereo
#   nix-shell            # first run creates .venv and pip-installs requirements.txt (~3 GB, PyTorch + CUDA wheels)
#
# PyTorch comes from pip wheels (they bundle CUDA), not nixpkgs. They run on NixOS through nix-ld
# (programs.nix-ld.enable in /etc/nixos/configuration.nix). The GPU driver comes from WSL (/usr/lib/wsl/lib).
# The native libraries below are what the opencv / open3d wheels need beyond nix-ld's defaults.
{ pkgs ? import <nixpkgs> { } }:

let
  python = pkgs.python312;
  libs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    libGL
    libglvnd
    glib
    libxkbcommon
    libx11
    libxext
    libxrender
    libxi
    libxrandr
    libxcursor
    libxinerama
    libxxf86vm
    libsm
    libice
    expat
    udev # libudev.so.1 (open3d)
    libusb1 # open3d (RealSense support)
    libidn2
    tbb # libtbb.so.12 (open3d)
  ];
in
pkgs.mkShell {
  packages = [ python pkgs.uv ];

  shellHook = ''
    export FS_DIR="${toString ./.}"
    # /usr/lib/wsl/lib: libcuda.so from the Windows NVIDIA driver (also makes nvidia-smi work)
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${pkgs.lib.makeLibraryPath libs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export UV_PYTHON="${python}/bin/python3"
    export UV_PYTHON_DOWNLOADS=never
    export UV_HTTP_TIMEOUT=300 # the multi-GB CUDA wheels time out at the default 30 s

    # (Re)install when requirements.txt is newer than the last successful install.
    if [ ! -f "$FS_DIR/.venv/.installed" ] || [ "$FS_DIR/requirements.txt" -nt "$FS_DIR/.venv/.installed" ]; then
      echo "Installing requirements.txt into $FS_DIR/.venv (first run downloads ~3 GB)..."
      [ -x "$FS_DIR/.venv/bin/python" ] || uv venv "$FS_DIR/.venv"
      uv pip install --python "$FS_DIR/.venv/bin/python" --index-strategy unsafe-best-match -r "$FS_DIR/requirements.txt" \
        && touch "$FS_DIR/.venv/.installed"
    fi
    source "$FS_DIR/.venv/bin/activate"
  '';
}
