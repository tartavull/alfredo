{ buildPythonPackage
, fetchFromGitHub
, absl-py
, etils
, jax
, jaxlib
, numpy
, tensorflow
, flit
, tensorstore
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
    etils
    jax
    jaxlib
    numpy
    tensorflow
    flit
    tensorstore
  ];
}
