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
  #, tensorstore
}:

buildPythonPackage rec {
  name = "orbax";
  src = fetchFromGitHub {
    owner = "google";
    repo = "orbax";
    rev = "v0.1.6";
    hash = "sha256-Vkqt2ovTan6bQJI4Il06hG0NlYmt60to4ue4U9qG9HY=";
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
    # tensorstore
  ];

  postPatch = ''
    sed -i '/tensorstore >= 0.1.20/d' pyproject.toml
    substituteInPlace pyproject.toml \
      --replace "'jax >= 0.4.6'," "'jax',"
  '';
}
