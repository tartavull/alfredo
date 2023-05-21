{ buildPythonPackage
}:

buildPythonPackage rec {
  name = "alfredo";
  src = ../alfredo/.;
  propagatedBuildInputs = [ ];
}
