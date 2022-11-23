{ lib
, buildPythonPackage
, fetchFromGitHub
, numpy
, cython
, fasteners
, glfw
, cffi
, mujoco
, mesa
, pkg-config
}:

buildPythonPackage rec {
  pname = "mujoco-py";
  version = "2.1.2.14";

  src = fetchFromGitHub {
    owner = "openai";
    repo = pname;
    rev = "v${version}";
    hash = "sha256-nwIJzLPhTZNlwk/NAiWCV/zYdwKeuQqhW6UIniGw8+k=";
  };

  nativeBuildInputs = [
    cython
    mesa
    pkg-config
  ];

  propagatedBuildInputs = [
    cython
    numpy
    fasteners
    cffi
    mujoco
    glfw
    mesa
    pkg-config
  ];

  MUJOCO_PY_MUJOCO_PATH="${mujoco}";
  LD_LIBRARY_PATH="${mujoco}/include:${mujoco}/bin";
  CPATH="-I ${mujoco}/include/mujoco";
  C_INCLUDE_PATH="-I ${mujoco}/include/mujoco";
  CPLUS_INCLUDE_PATH="-I ${mujoco}/include/mujoco";


  meta = with lib; {
    license = licenses.mit;
  };
}
