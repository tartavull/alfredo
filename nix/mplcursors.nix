{ lib
, buildPythonPackage
, fetchPypi
, setuptools_scm
, pytest
, pytestCheckHook
, matplotlib
}:

buildPythonPackage rec {
  pname = "mplcursors";
  version = "0.5";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-w92Ej+z0b7xIdAikBgouEDpVZ8EERN8OrIswjEBU5U0=";
  };

  nativeBuildInputs = [ setuptools_scm ];
  propagatedBuildInputs = [ matplotlib ];
  checkInputs = [ pytest pytestCheckHook ];

  doCheck = false;

  meta = with lib; {
    description = "Interactive data selection cursors for Matplotlib";
    homepage = "https://github.com/anntzer/mplcursors";
    license = licenses.mit;
    maintainers = with maintainers; [ ];
  };
}
