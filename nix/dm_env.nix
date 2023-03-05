{ buildPythonPackage
, fetchFromGitHub
, pytest
, absl-py
, dm-tree
, numpy
, nose
}:

buildPythonPackage rec {
  name = "dm_env";
  src = fetchFromGitHub {
    owner = "deepmind";
    repo = "dm_env";
    rev = "v1.5";
    hash = "sha256-aEudcqmrLquZ9XvK/7pjyLvN/nDIeuYFQjdD1Cyh3Us=";
  };
  propagatedBuildInputs = [
    pytest
    absl-py
    dm-tree
    numpy
    nose
  ];
}
