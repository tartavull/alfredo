{ buildPythonPackage
, lib
, absl-py
, cached-property
, etils
, fetchPypi
, flit
, jax
, jaxlib
, msgpack
, nest-asyncio
, numpy
, pyyaml
, tensorflow
, tensorstore
}:

buildPythonPackage rec {
  pname = "orbax-checkpoint";
  version = "0.1.6";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-lnh2eAr54Dk8C9hnW/nZP0vrz/9Vqvwo5FMfrkJqFsA=";
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
    tensorstore
  ];

  meta = with lib; {
    description = "Checkpointing library for JAX-based models";
    license = licenses.asl20;
    homepage = "https://github.com/google/orbax";
  };
}
