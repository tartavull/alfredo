{ buildPythonPackage
, fetchFromGitHub
, jax
, jaxlib
, orbax
, rich
, optax
, matplotlib
}:

buildPythonPackage rec {
  name = "flax";
  src = fetchFromGitHub {
    owner = "google";
    repo = "flax";
    rev = "v0.6.5";
    hash = "sha256-Vv68BK83gTIKj0r9x+twdhqmRYziD0vxQCdHkYSeTak=";
  };
  propagatedBuildInputs = [
    jax
    jaxlib
    orbax
    rich
    optax
    matplotlib
  ];
  postPatch = ''
    sed -i '/tensorstore/d' setup.py
  '';
  doCheck = false;
  pythonRemoveDeps = [ "orbax" ];
}
