# { buildPythonPackage
# , fetchFromGitHub
# , setuptools
# , pkgs
# }:

# buildPythonPackage rec {
#   name = "tensorstore";
  
#   src = fetchFromGitHub {
#     owner = "google";
#     repo = "tensorstore";
#     rev = "v0.1.60";
#     hash = "sha256-rT0R1x51xHAElPwernUjBIIneRhncnsohMRAIhXyaYk=";
#   };
#   format = "pyproject";

#   buildinputs = [
#     pkgs.bazel
#   ];
  
#   propagatedBuildInputs = [
#     setuptools
#   ];

#   preConfigure = ''
#     export HOME=$PWD
#   '';

# }

{ buildPythonPackage
, fetchPypi
, pkgs
, pip
, lib
, setuptools
}:

buildPythonPackage rec {
  pname = "tensorstore";
  version = "0.1.6";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-MtzIVk6oCmF0gSFisrruxSUyNdVwIciMZNY0sORN1Jg=";
  };

  propagatedBuildInputs = [
    pip
    setuptools
  ];

  # nativeBuildInputs = [pkgs.python3Packages.pip];

  meta = with lib; {
    description = "TensorStore";
    license = licenses.asl20;
    homepage = "https://github.com/google/tensorstore";
  };  
}
