{ buildPythonPackage
, fetchFromGitHub
, pytest
}:

buildPythonPackage rec {
  name = "genetic-intelligence";
  src = ../.;
  propagatedBuildInputs = [ pytest ];
}
