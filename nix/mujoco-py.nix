{ lib
, buildPythonPackage
, fetchFromGitHub
, numpy
, cython
, fasteners
, glfw
, cffi
, mujoco
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

  propagatedBuildInputs = [
    cython
    numpy
    fasteners
    cffi
    mujoco
  ];

  MUJOCO_PY_MUJOCO_PATH="${mujoco}";
  LD_LIBRARY_PATH="${mujoco}/lib:${mujoco}/bin";
  CPATH="-I${mujoco}/include/mujoco";
  C_INCLUDE_PATH="-I${mujoco}/include/mujoco";
  CPLUS_INCLUDE_PATH="-I${mujoco}/include/mujoco";


  meta = with lib; {
    license = licenses.mit;
    maintainers = with maintainers; [ tartavull ];
  };
}
