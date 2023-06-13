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
    rev = "91b46797fea731f80eab8cd2c8352a0674141d89";
    hash = "sha256-yvvj8kKH+8afYPOJqJ1mBdjwxaiLdF4YVZo76CeZVFc=";
  };
  propagatedBuildInputs = [
    pytest
    absl-py
    dm-tree
    numpy
    nose
  ];
}
