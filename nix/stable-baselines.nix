{ lib
, buildPythonPackage
, fetchFromGitHub
, pythonRelaxDepsHook
, pytest
, numpy
, pandas
, gym
, matplotlib
, joblib
, pytorch
, importlib-metadata
}:

buildPythonPackage rec {
  name = "stable-baselines";
  src = fetchFromGitHub {
    owner = "DLR-RM";
    repo = "stable-baselines3";
    rev = "d5d1a02c15cdce868c72bbc94913e66fdd2efd3a";
    hash = "sha256-zCAe5mfqyVlnfepiYwISUTSdlscBaSW7BZ4R+0kaHYE=";
  };
  format = "setuptools";
  prePatch = ''
    substituteInPlace setup.py \
        --replace 'importlib-metadata~=4.13' 'importlib-metadata' \
        --replace 'gym==0.21' 'gym'
  '';
  doCheck = false;
  # unfortuneatly this package depends on the now agent Tensorflow 1 which had
  # tensorflow.contrib Until this package supports tensorflow 2 we won't be
  # able to run it because tensorflow.contrib is in another package.

  nativeBuildInputs = [ pythonRelaxDepsHook ];
  pythonRelaxDeps = true;
  propagatedBuildInputs = [
    pytest
    numpy
    pandas
    gym
    matplotlib
    joblib
    pytorch
    importlib-metadata
  ];

}
