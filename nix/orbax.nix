{ buildPythonPackage
, absl-py
, cached-property
, etils
, fetchFromGitHub
, flit
, jax
, jaxlib
, msgpack
, nest-asyncio
, numpy
, pyyaml
, tensorflow
, importlib-resources
  #, tensorstore
}:

buildPythonPackage rec {
  name = "orbax-checkpoint";
  src = fetchFromGitHub {
    owner = "google";
    repo = "orbax";
    rev = "v0.1.7";
    hash = "sha256-Zk9hbvSA82jt0wLR7AZWEmHDA4A1+9t0ezf74FYkqe0=";
  };
  format = "pyproject";

  propagatedBuildInputs = [
    absl-py
    cached-property
    etils
    flit
    jax
    jaxlib
    msgpack
    nest-asyncio
    numpy
    pyyaml
    tensorflow
    importlib-resources
    # tensorstore
  ];

  postPatch = ''
    sed -i '/tensorstore >= 0.1.20/d' pyproject.toml
    substituteInPlace pyproject.toml \
      --replace "'jax >= 0.4.6'," "'jax',"
  '';
}
