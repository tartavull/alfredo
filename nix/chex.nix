{ absl-py
, buildPythonPackage
, dm-tree
, fetchFromGitHub
, jax
, jaxlib
, lib
, numpy
, pytestCheckHook
, toolz
, cloudpickle
, typing-extensions
}:

buildPythonPackage rec {
  pname = "chex";
  version = "0.1.7";
  format = "setuptools";

  src = fetchFromGitHub {
    owner = "deepmind";
    repo = pname;
    rev = "refs/tags/v${version}";
    hash = "sha256-mz9X/5k71sBSlh4tT81tD0oE25E0O17WNEtXq/wvYAA=";
  };

  propagatedBuildInputs = [
    absl-py
    cloudpickle
    dm-tree
    jax
    numpy
    toolz
    typing-extensions
  ];

  postPatch = ''
    substituteInPlace requirements/requirements.txt \
      --replace "jax>=0.4.6" "jax"
  '';

  pythonImportsCheck = [
    "chex"
  ];

  checkInputs = [
    jaxlib
    pytestCheckHook
  ];

  disabledTests = [
    # See https://github.com/deepmind/chex/issues/204.
    "test_uninspected_checks"
  ];

  meta = with lib; {
    description = "Chex is a library of utilities for helping to write reliable JAX code.";
    homepage = "https://github.com/deepmind/chex";
    license = licenses.asl20;
    maintainers = with maintainers; [ ndl ];
  };
}
