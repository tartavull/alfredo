{ absl-py
, buildPythonPackage
, chex
, fetchFromGitHub
, jax
, jaxlib
, lib
, numpy
, callPackage
}:

buildPythonPackage rec {
  pname = "optax";
  version = "0.1.5";

  src = fetchFromGitHub {
    owner = "deepmind";
    repo = pname;
    rev = "refs/tags/v${version}";
    hash = "sha256-vhPpynKq0dboSt+fQ4lvVv9ytDXnZKRrc7lF03Mm39g=";
  };

  outputs = [
    "out"
    "testsout"
  ];

  buildInputs = [ jaxlib ];

  propagatedBuildInputs = [
    absl-py
    chex
    numpy
    jaxlib
    jax
  ];

  postInstall = ''
    mkdir $testsout
    cp -R examples $testsout/examples
  '';

  pythonImportsCheck = [
    "optax"
  ];

  # check in passthru.tests.pytest to escape infinite recursion with flax
  doCheck = false;

  passthru.tests = {
    pytest = callPackage ./tests.nix { };
  };

  meta = with lib; {
    description = "Optax is a gradient processing and optimization library for JAX.";
    homepage = "https://github.com/deepmind/optax";
    license = licenses.asl20;
    maintainers = with maintainers; [ ndl ];
  };
}
