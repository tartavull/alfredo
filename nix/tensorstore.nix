{ buildPythonPackage
, fetchFromGitHub
, setuptools
, pkgs
}:

buildPythonPackage rec {
  name = "tensorstore";
  
  src = fetchFromGitHub {
    owner = "google";
    repo = "tensorstore";
    rev = "v0.1.60";
    hash = "sha256-rT0R1x51xHAElPwernUjBIIneRhncnsohMRAIhXyaYk=";
  };
  format = "pyproject";

  buildinputs = [
    pkgs.bazel
  ];
  
  propagatedBuildInputs = [
    setuptools
  ];

  preConfigure = ''
    export HOME=$PWD
  '';
  
}
