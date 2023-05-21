{ buildPythonPackage
, fetchFromGitHub
, setuptools
}:

buildPythonPackage rec {
  name = "tensorstore";
  src = fetchFromGitHub {
    owner = "google";
    repo = "tensorstore";
    rev = "v0.1.35";
    hash = "sha256-VmJHDoU+lDS3PT4cEDZVDY+VYTa0F2X9aUWIEZW29vM=";
  };
  format = "pyproject";
  propagatedBuildInputs = [
    setuptools
  ];
}
