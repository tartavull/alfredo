{ buildPythonPackage
}:

buildPythonPackage rec {
  name = "alfredo";
  src = ../src/.;
  propagatedBuildInputs = [ ];
}
