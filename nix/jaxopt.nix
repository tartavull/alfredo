{ lib
, buildPythonPackage
, numpy
, scipy
, jax
, jaxlib
, pytest
, fetchFromGitHub
, matplotlib
, scikit-learn
}:

buildPythonPackage rec {
  pname = "jaxopt";
  version = "0.6";

  src = fetchFromGitHub {
    owner = "google";
    repo = pname;
    rev = "${pname}-v${version}";
    hash = "sha256-hRMVBvEl2wqJqyRsY1f/XqBp6OA1vi18i69nuE3YbqE=";
  };

  propagatedBuildInputs = [
    numpy
    scipy
    jax
    matplotlib
    scikit-learn
  ];

  doCheck = true;
  checkInputs = [
    pytest
    jaxlib
  ];

  meta = with lib; {
    description = "Hardware accelerated, batchable and differentiable optimizers in JAX.";
    homepage = "https://github.com/google/jaxopt";
    maintainers = with maintainers; [ tartavull ];
    platforms = platforms.all;
  };
}
