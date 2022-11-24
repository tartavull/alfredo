{ fetchFromGitHub
, mesa
, python3
, libGL
, gcc
, stdenv
, callPackage
, autoPatchelfHook
# , xorg
, lib
# , libglvnd
, imageio
, numpy
, cython
, cffi
, lockfile
, buildPythonPackage
# , sources
, fasteners
, mujoco
, pkgs
, writeText
, glfw
, pkg-config
}:
buildPythonPackage rec {
  pname = "mujoco-py";
  version = "2.1.2.14";

  src = fetchFromGitHub {
    owner = "openai";
    repo = "mujoco-py";
    rev = "v${version}";
    hash = "sha256-nwIJzLPhTZNlwk/NAiWCV/zYdwKeuQqhW6UIniGw8+k=";
  };

  python = python3;

  nativeBuildInputs = [
    autoPatchelfHook
    pkg-config
  ];
  propagatedBuildInputs = [
    imageio
    numpy
    cython
    glfw
    cffi
    lockfile
    fasteners
    mujoco
  ];
  buildInputs = [
    mesa
    mesa.osmesa
    mujoco
    python3
    libGL
    gcc
    stdenv.cc.cc.lib
  ];

  LD_LIBRARY_PATH="${mujoco}/include:${mujoco}/bin";
  CPATH="-I ${mujoco}/include/mujoco";
  C_INCLUDE_PATH="-I ${mujoco}/include/mujoco";
  CPLUS_INCLUDE_PATH="-I ${mujoco}/include/mujoco";

  #   doCheck = false;
}
