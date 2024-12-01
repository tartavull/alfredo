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
, setuptools_scm
}:

buildPythonPackage rec {
  pname = "tensorstore";
  version = "0.1.6";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-MtzIVk6oCmF0gSFisrruxSUyNdVwIciMZNY0sORN1Jg=";
  };

  nativeBuiltInputs = [
    pkgs.python3Packages.setuptools_scm
    # (python.pkgs.setuptools.override { version = "67.0.0"; })
  ];

  propagatedBuildInputs = [
    # pip
    setuptools
  ];

  # patchPhase = ''
  #   sed -i 's/class BuildCommand(/from setuptools import Command\nclass BuildCommand(Command)/' setup.py
  #   sed -i '/cmdclass/,/},/d' setup.py
  # '';
  
  # patchPhase = ''
  #   sed -i 's/class BuildCommand:/from setuptools import Command\nclass BuildCommand(Command):/' setup.py
  #   # Optional: Force setuptools_scm to write the version file
  #   echo "[tool.setuptools_scm]" >> pyproject.toml
  #   echo "write_to = \"tensorstore/_version.py\"" >> pyproject.toml
  # '';

  # # Fix issues with setuptools_scm
  # patchPhase = ''
  #   if [ ! -f pyproject.toml ]; then
  #     echo "[tool.setuptools_scm]" >> pyproject.toml
  #     echo "write_to = \"tensorstore/_version.py\"" >> pyproject.toml
  #   fi
  # '';

  # # Build with a writable $HOME directory
  # buildPhase = ''
  #   export SETUPTOOLS_SCM_PRETEND_VERSION="0.1.6"
  #   export HOME=$TMPDIR
  #   python setup.py build
  # '';

  meta = with lib; {
    description = "TensorStore";
    license = licenses.asl20;
    homepage = "https://github.com/google/tensorstore";
  };  
}
